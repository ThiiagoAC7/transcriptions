from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

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
    input_dir: str,
    videos_dir: str,
    output_dir: str,
    speakers: bool = True,
    youtubers: List[str] | None = None,
) -> None:
    """
    Extract transcriptions from all audio files in the input directory.
    Processes in batches: download up to 5 videos, transcribe them,
    then delete the raw audio files to save disk space.

    params:
    - input_dir: directory containing audio files
    - output_dir: directory to save transcription files
    - speakers: enable speaker diarization
    - youtubers: optional list of youtubers to filter

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Speaker diarization: {'enabled' if speakers else 'disabled'}")

    os.makedirs(output_dir, exist_ok=True)

    all_video_ids = collect_video_ids(videos_dir)
    all_ytbrs = os.listdir(input_dir)
    ytbrs = all_ytbrs if not youtubers else [y for y in youtubers if y in all_ytbrs]

    for y in ytbrs:
        audio_dir = os.path.join(input_dir, y)
        output_ytbr_dir = os.path.join(output_dir, y)
        os.makedirs(output_ytbr_dir, exist_ok=True)

        # process any already-downloaded audio files first
        if os.path.exists(audio_dir):
            existing_audios = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        else:
            existing_audios = []
            os.makedirs(audio_dir, exist_ok=True)

        if existing_audios:
            print(f"found {len(existing_audios)} existing audio files for {y}")
            for i, audio_file in enumerate(existing_audios):
                print(f"[{i + 1}/{len(existing_audios)}] processing {y}/{audio_file}...")
                file_path = os.path.join(audio_dir, audio_file)
                name = os.path.splitext(audio_file)[0]
                process_audio_file(file_path, y, name, device, output_dir, speakers)

            _delete_audio_files(audio_dir, existing_audios)
            print(f"deleted {len(existing_audios)} processed audio files for {y}")

        # determine remaining video IDs that need transcription
        video_ids_for_y = all_video_ids.get(y, [])
        if not video_ids_for_y:
            print(f"No video IDs found for {y}")
            continue

        remaining_ids = [
            vid for vid in video_ids_for_y
            if not os.path.exists(os.path.join(output_ytbr_dir, f"{vid}.txt"))
            and not os.path.exists(os.path.join(output_ytbr_dir, f"{vid}.json"))
        ]

        if not remaining_ids:
            print(f"all videos already transcribed for {y}")
            continue

        total_batches = (len(remaining_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"{len(remaining_ids)} videos remaining for {y} ({total_batches} batches)")

        for batch_start in range(0, len(remaining_ids), BATCH_SIZE):
            batch_ids = remaining_ids[batch_start:batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1

            print(f"\n--- batch {batch_num}/{total_batches} for {y}: {len(batch_ids)} videos ---")

            download_videos({y: batch_ids}, input_dir)

            # process each downloaded audio file in the batch
            processed_files: List[str] = []
            for vid in batch_ids:
                audio_path = os.path.join(audio_dir, f"{vid}.wav")
                if os.path.exists(audio_path):
                    print(f"[{len(processed_files) + 1}/{len(batch_ids)}] Processing {y}/{vid}...")
                    process_audio_file(audio_path, y, vid, device, output_dir, speakers)
                    processed_files.append(f"{vid}.wav")
                else:
                    print(f"[WARN] Expected audio file not found: {audio_path}")

            # delete raw audio files to free disk space
            _delete_audio_files(audio_dir, processed_files)
            print(f"Batch {batch_num} complete. Deleted {len(processed_files)} audio files.")

        print(f"\nAll batches complete for {y}")


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
