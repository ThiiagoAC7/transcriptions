from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from dotenv import load_dotenv

from download import collect_video_ids, download_videos
from speech import (
    TranscriptionModelCache,
    format_plain_text,
    format_speakers_to_txt,
    transcribe_old,
    transcribe_with_speakers,
)

load_dotenv()

VideoIdMap = Dict[str, List[str]]

BATCH_SIZE = 5


def _delete_audio_files(audio_dir: str, filenames: List[str]) -> None:
    for filename in filenames:
        path = os.path.join(audio_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def process_audio_file(
    audio_path: str,
    youtuber: str,
    name: str,
    device: str,
    output_dir: str,
    speakers: bool = True,
    model_cache: TranscriptionModelCache | None = None,
) -> None:
    """
    Process an audio file and save transcription results.

    params:
    - audio_path: path to the audio file
    - youtuber: name of the youtuber
    - name: base filename for output
    - device: torch device to use for inference
    - output_dir: directory to save transcription files
    - speakers: enable speaker diarization
    - model_cache: optional model cache
    """
    txt_path = os.path.join(output_dir, youtuber, f"{name}.txt")
    json_path = os.path.join(output_dir, youtuber, f"{name}.json")

    if speakers:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print(f"[SKIP] HF_TOKEN not set, falling back to old pipeline")
            speakers = False
        elif os.path.exists(json_path):
            print(f"[SKIP] {youtuber}/{name}.json already exists.")
            return

    if speakers:
        try:
            result = transcribe_with_speakers(
                audio_path, hf_token, device, model_cache=model_cache
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            txt_content = format_speakers_to_txt(result)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)

        except Exception as e:
            print(f"  ❌ Speaker diarization failed: {e}")
            print(f"[SKIP] Skipping {name} due to diarization failure.")
            return

    if not speakers:
        if os.path.exists(txt_path):
            print(f"[SKIP] {youtuber}/{name}.txt already exists.")
            return

        result = transcribe_old(audio_path, device, model_cache=model_cache)
        txt_content = format_plain_text(result)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)



def _process_audio_files(
    audio_dir: str,
    youtuber: str,
    audio_filenames: List[str],
    device: str,
    output_dir: str,
    speakers: bool,
    model_cache: TranscriptionModelCache | None = None,
) -> None:
    """
    processa e deleta uma lista de arquivos de audio de um youtuber.

    params:
    - audio_dir: diretorio contendo arquivos de audio
    - youtuber: nome do youtuber
    - audio_filenames: lista de nomes de arquivos de audio
    - device: dispositivo torch a ser usado
    - output_dir: diretorio para salvar transcricoes
    - speakers: habilitar diarizacao de locutores
    - model_cache: cache de modelo opcional
    """
    processed: List[str] = []
    total = len(audio_filenames)
    for idx, filename in enumerate(audio_filenames):
        path = os.path.join(audio_dir, filename)
        if not os.path.exists(path):
            print(f"[warn] arquivo de audio esperado nao encontrado: {path}")
            continue

        name = os.path.splitext(filename)[0]
        print(f"[{idx + 1}/{total}] processando {youtuber}/{name}...")
        process_audio_file(
            path,
            youtuber,
            name,
            device,
            output_dir,
            speakers,
            model_cache=model_cache,
        )
        processed.append(filename)

    if processed:
        _delete_audio_files(audio_dir, processed)
        print(f"apagados {len(processed)} arquivos de audio processados para {youtuber}")


def _process_existing_audios(
    audio_dir: str,
    youtuber: str,
    device: str,
    output_dir: str,
    speakers: bool,
    model_cache: TranscriptionModelCache,
) -> None:
    """
    processa arquivos de audio ja existentes para um youtuber.

    params:
    - audio_dir: diretorio contendo arquivos de audio
    - youtuber: nome do youtuber
    - device: dispositivo torch a ser usado
    - output_dir: diretorio para salvar transcricoes
    - speakers: habilitar diarizacao de locutores
    - model_cache: cache de modelo
    """
    if os.path.exists(audio_dir):
        existing_audios = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    else:
        existing_audios = []
        os.makedirs(audio_dir, exist_ok=True)

    if existing_audios:
        print(f"encontrados {len(existing_audios)} arquivos de audio existentes para {youtuber}")
        _process_audio_files(
            audio_dir=audio_dir,
            youtuber=youtuber,
            audio_filenames=existing_audios,
            device=device,
            output_dir=output_dir,
            speakers=speakers,
            model_cache=model_cache,
        )


def _process_batches(
    remaining_ids: List[str],
    youtuber: str,
    audio_dir: str,
    input_dir: str,
    output_dir: str,
    device: str,
    speakers: bool,
    model_cache: TranscriptionModelCache,
) -> None:
    """
    processa os ids de video restantes em lotes.

    params:
    - remaining_ids: lista de ids de video restantes
    - youtuber: nome do youtuber
    - audio_dir: diretorio contendo arquivos de audio
    - input_dir: diretorio de entrada contendo audios
    - output_dir: diretorio para salvar transcricoes
    - device: dispositivo torch a ser usado
    - speakers: habilitar diarizacao de locutores
    - model_cache: cache de modelo
    """
    total_batches = (len(remaining_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"{len(remaining_ids)} videos restantes para {youtuber} ({total_batches} lotes)")

    for batch_start in range(0, len(remaining_ids), BATCH_SIZE):
        batch_ids = remaining_ids[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1

        print(f"\n--- lote {batch_num}/{total_batches} para {youtuber}: {len(batch_ids)} videos ---")

        download_videos({youtuber: batch_ids}, input_dir)

        batch_filenames = [f"{vid}.wav" for vid in batch_ids]
        _process_audio_files(
            audio_dir=audio_dir,
            youtuber=youtuber,
            audio_filenames=batch_filenames,
            device=device,
            output_dir=output_dir,
            speakers=speakers,
            model_cache=model_cache,
        )
        print(f"lote {batch_num} completo para {youtuber}")


def extract_transcriptions(
    input_dir: str,
    videos_dir: str,
    output_dir: str,
    speakers: bool = True,
    youtubers: List[str] | None = None,
) -> None:
    """
    extrai transcricoes de todos os arquivos de audio no diretorio de entrada.

    params:
    - input_dir: diretorio contendo arquivos de audio
    - videos_dir: diretorio contendo metadados dos videos
    - output_dir: diretorio para salvar transcricoes
    - speakers: habilitar diarizacao de locutores
    - youtubers: lista opcional de youtubers para filtrar
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"dispositivo usado: {device}")
    print(f"diarizacao de locutores: {'ativada' if speakers else 'desativada'}")

    os.makedirs(output_dir, exist_ok=True)

    all_video_ids = collect_video_ids(videos_dir)
    all_ytbrs = os.listdir(input_dir)
    ytbrs = all_ytbrs if not youtubers else [y for y in youtubers if y in all_ytbrs]

    hf_token = os.getenv("HF_TOKEN")
    model_cache = TranscriptionModelCache(device=device, hf_token=hf_token)

    try:
        for y in ytbrs:
            audio_dir = os.path.join(input_dir, y)
            output_ytbr_dir = os.path.join(output_dir, y)
            os.makedirs(output_ytbr_dir, exist_ok=True)

            # processa arquivos ja baixados
            _process_existing_audios(
                audio_dir=audio_dir,
                youtuber=y,
                device=device,
                output_dir=output_dir,
                speakers=speakers,
                model_cache=model_cache,
            )

            # determina ids de video restantes que precisam de transcricao
            video_ids_for_y = all_video_ids.get(y, [])
            if not video_ids_for_y:
                print(f"nenhum id de video encontrado para {y}")
                continue

            remaining_ids = [
                vid for vid in video_ids_for_y
                if not os.path.exists(os.path.join(output_ytbr_dir, f"{vid}.txt"))
                and not os.path.exists(os.path.join(output_ytbr_dir, f"{vid}.json"))
            ]

            if not remaining_ids:
                print(f"todos os videos ja transcritos para {y}")
                continue

            _process_batches(
                remaining_ids=remaining_ids,
                youtuber=y,
                audio_dir=audio_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                device=device,
                speakers=speakers,
                model_cache=model_cache,
            )

            print(f"\ntodos os lotes concluidos para {y}")
    finally:
        model_cache.clear()



def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube video transcription pipeline")
    parser.add_argument(
        "--input-dir",
        default="downloads",
        help="Directory containing audio files (default: downloads)",
    )
    parser.add_argument(
        "--output-dir",
        default="text",
        help="Directory to save transcriptions (default: text)",
    )
    parser.add_argument(
        "--videos-dir",
        default="videos",
        help="Directory containing selected videos (default: videos)",
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        default=True,
        help="Enable speaker diarization (default: True)",
    )
    parser.add_argument(
        "--no-speakers",
        action="store_false",
        dest="speakers",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--youtubers",
        nargs="*",
        default=[],
        help="Filter to specific youtubers (default: all)",
    )
    args = parser.parse_args()

    extract_transcriptions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        videos_dir=args.videos_dir,
        speakers=args.speakers,
        youtubers=args.youtubers,
    )


if __name__ == "__main__":
    main()
