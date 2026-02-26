import argparse
import gc
import os
import json

import torch
import yt_dlp
from transformers import pipeline
from dotenv import load_dotenv

import random
import time

import whisperx

load_dotenv()


def transcribe_old(audio_path: str, device: str) -> dict:
    """Executa transcrição com transformers Whisper (sem diarização)."""
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
        chunk_length_s=30,
        batch_size=24,
    )

    result = pipe(audio_path, return_timestamps=True, language="en")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return result


def transcribe_with_speakers(
    audio_path: str,
    hf_token: str,
    device: str,
    max_speakers: int = 5
) -> dict:
    """Transcreve áudio com WhisperX e diarização de speaker."""
    model = whisperx.load_model("large-v3-turbo", device)

    result = model.transcribe(audio_path, language="en")

    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio_path, device)

    del model_a
    torch.cuda.empty_cache()

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio_path, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    del model, diarize_model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def format_plain_text(result: dict) -> str:
    """Extrai texto puro do resultado da transcrição."""
    return result.get("text", "").strip()


def format_speakers_to_txt(result: dict) -> str:
    """Converte resultado WhisperX para formato [SPEAKER_X] texto."""
    lines = []
    for segment in result.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


def process_audio_file(
    audio_path: str,
    youtuber: str,
    name: str,
    device: str,
    speakers: bool = True
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


def extract_transcriptions(speakers: bool = True) -> None:
    """Extrai transcrições de todos os áudios em ./downloads."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Speaker diarization: {'enabled' if speakers else 'disabled'}")

    os.makedirs("./text/", exist_ok=True)

    ytbrs = os.listdir('./downloads')

    for y in ytbrs:
        audio_dir = f"./downloads/{y}"
        os.makedirs(f"./text/{y}", exist_ok=True)

        if os.path.exists(audio_dir):
            audios = [f for f in os.listdir(audio_dir) if f.endswith(('.wav'))]
        else:
            audios = []
            print(f"Directory {audio_dir} does not exist.")

        for i, audio_file in enumerate(audios):
            print(f"[{i+1}/{len(audios)}] Processing {y}/{audio_file}...")
            file_path = os.path.join(audio_dir, audio_file)
            name = os.path.splitext(audio_file)[0]

            process_audio_file(file_path, y, name, device, speakers)


def collect_video_ids(base_dir="transcriptions/videos"):
    """
    Traverses the base_dir to find 'most_viewed_videos_per_month.json' files
    and collects all video IDs from the 'selected_videos' field.
    """
    video_ids = {}
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "most_viewed_videos_per_month.json":
                file_path = os.path.join(root, file)
                video_ids[os.path.basename(root)] = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    selected = data.get("selected_videos", [])
                    video_ids[os.path.basename(root)].extend(selected)
                    print(f"Loaded {len(selected)} videos from {os.path.basename(root)}")

    return video_ids


def download_videos(video_ids={}):
    """
    downloads audio from specific video_ids
    using .wav for best quality

    params:
        - video_ids: list of video ids as they appear in the video url
    """
    if not video_ids:
        print("no video IDs provided.")
        return

    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0'

    base_opts = {
        'format': 'bestaudio/best',
        'quiet': False,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        # 'cookies_from_browser': 'firefox',
        'cookiefile' : 'cookies.txt',
        'user_agent': user_agent,
        'retries': 10,
        'fragment_retries': 10,
        'retry_sleep': 20,
        'sleep_interval': 10,
        'max_sleep_interval': 30,
        'source_address': '0.0.0.0',
        'http_headers': {
            'Referer': 'https://www.google.com/',
            'Accept-Language': 'en-US,en;q=0.9',
        },
    }

    for ytbr, vids in video_ids.items():
        output_dir = os.path.join('transcriptions', 'downloads', ytbr)

        urls_to_download = []

        for vid in vids:
            expected_path = os.path.join(output_dir, f"{vid}.wav")

            if os.path.exists(expected_path):
                print(f"[SKIP] {ytbr}/{vid} already exists.")
            else:
                urls_to_download.append(f"https://www.youtube.com/watch?v={vid}")

        if urls_to_download:
            print(f"Downloading {len(urls_to_download)} new videos for {ytbr}...")

            current_opts = base_opts.copy()
            current_opts['outtmpl'] = f'{output_dir}/%(id)s.%(ext)s'

            with yt_dlp.YoutubeDL(current_opts) as ydl:
                ydl.download(urls_to_download)

            print("Batch finished. Resting for to cool down...")
            time.sleep(random.randint(30, 60))
        else:
            print(f"All videos for {ytbr} are up to date.")


def main():
    parser = argparse.ArgumentParser(description="YouTube video transcription pipeline")
    parser.add_argument(
        "--speakers",
        action="store_true",
        default=True,
        help="Enable speaker diarization (default: True)"
    )
    parser.add_argument(
        "--no-speakers",
        action="store_false",
        dest="speakers",
        help="Disable speaker diarization"
    )
    args = parser.parse_args()

    extract_transcriptions(speakers=args.speakers)


if __name__ == "__main__":
    main()
