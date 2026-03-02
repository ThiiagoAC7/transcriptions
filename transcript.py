from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from dotenv import load_dotenv

from download import collect_video_ids, download_videos
from speech import (
    format_plain_text,
    format_speakers_to_txt,
    transcribe_old,
    transcribe_with_speakers,
)

load_dotenv()


def process_audio_file(
    audio_path: str, youtuber: str, name: str, device: str, speakers: bool = True
) -> None:
    """Processa um arquivo de áudio e salva os resultados."""
    txt_path = f"./text/{youtuber}/{name}.txt"
    json_path = f"./text/{youtuber}/{name}.json"

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
            result = transcribe_with_speakers(audio_path, hf_token, device)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            txt_content = format_speakers_to_txt(result)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)

        except Exception as e:
            print(f"Speaker diarization failed: {e}")
            print(f"[FALLBACK] Retrying with old pipeline...")
            speakers = False

    if not speakers:
        if os.path.exists(txt_path):
            print(f"[SKIP] {youtuber}/{name}.txt already exists.")
            return

        result = transcribe_old(audio_path, device)
        txt_content = format_plain_text(result)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)


def extract_transcriptions(
    speakers: bool = True, youtubers: List[str] | None = None
) -> None:
    """Extrai transcrições de todos os áudios em ./downloads."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Speaker diarization: {'enabled' if speakers else 'disabled'}")

    os.makedirs("./text/", exist_ok=True)

    all_video_ids = collect_video_ids()
    all_ytbrs = os.listdir("./downloads")
    ytbrs = all_ytbrs if not youtubers else [y for y in youtubers if y in all_ytbrs]

    for y in ytbrs:
        audio_dir = f"./downloads/{y}"
        os.makedirs(f"./text/{y}", exist_ok=True)

        if os.path.exists(audio_dir):
            audios = [f for f in os.listdir(audio_dir) if f.endswith((".wav"))]
        else:
            audios = []
            print(f"Directory {audio_dir} does not exist.")
            if y in all_video_ids and all_video_ids.get(y):
                print(f"Downloading videos for {y}...")
                download_videos({y: all_video_ids[y]})
                if os.path.exists(audio_dir):
                    audios = [f for f in os.listdir(audio_dir) if f.endswith((".wav"))]
                else:
                    print(f"No videos downloaded for {y}")
            else:
                print(f"No video IDs found for {y}")

        for i, audio_file in enumerate(audios):
            print(f"[{i + 1}/{len(audios)}] Processing {y}/{audio_file}...")
            file_path = os.path.join(audio_dir, audio_file)
            name = os.path.splitext(audio_file)[0]

            process_audio_file(file_path, y, name, device, speakers)


def main():
    parser = argparse.ArgumentParser(description="YouTube video transcription pipeline")
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

    extract_transcriptions(speakers=args.speakers, youtubers=args.youtubers)


if __name__ == "__main__":
    main()
