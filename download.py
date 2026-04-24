from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import yt_dlp


def collect_video_ids(base_dir: str) -> Dict[str, List[str]]:
    """
    Traverse base_dir to find 'most_viewed_videos_per_month.json' files
    and collect all video IDs from the 'selected_videos' field.

    params:
    - base_dir: base directory to search for metadata files

    returns:
    - dict mapping youtuber name to list of video IDs
    """
    video_ids: Dict[str, List[str]] = {}
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return video_ids

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "most_viewed_videos_per_month.json":
                file_path = os.path.join(root, file)
                youtuber = os.path.basename(root)
                video_ids[youtuber] = []
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    selected = data.get("selected_videos", [])
                    video_ids[youtuber].extend(selected)
                    print(f"Loaded {len(selected)} videos from {youtuber}")

    return video_ids


def download_videos(video_ids: Dict[str, List[str]], output_dir: str) -> None:
    """
    Download audio from specific video IDs using .wav for better quality.

    params:
    - video_ids: dict mapping youtuber to list of video IDs
    - output_dir: base directory to save downloaded audio files

    """
    if not video_ids:
        print("no video IDs provided.")
        return

    base_opts = {
        "format": "bestaudio/best",
        "quiet": False,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # 'cookies_from_browser': 'firefox',
        "cookiefile": "cookies.txt",
        # 'user_agent': user_agent,
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep": 20,
        "sleep_interval": 10,
        "max_sleep_interval": 30,
        "source_address": "0.0.0.0",
        "http_headers": {
            "Referer": "https://www.google.com/",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "js_runtimes": {"node": {}},
        "extractor_args": {
            "youtube": {
                "player_client": ["web"],
            }
        },
    }

    for ytbr, vids in video_ids.items():
        ytbr_output_dir = os.path.join(output_dir, ytbr)

        urls_to_download = []

        for vid in vids:
            expected_path = os.path.join(ytbr_output_dir, f"{vid}.wav")

            if os.path.exists(expected_path):
                print(f"[SKIP] {ytbr}/{vid} already exists.")
            else:
                urls_to_download.append(f"https://www.youtube.com/watch?v={vid}")

        if urls_to_download:
            print(f"Downloading {len(urls_to_download)} new videos for {ytbr}...")

            current_opts = base_opts.copy()
            current_opts["outtmpl"] = f"{ytbr_output_dir}/%(id)s.%(ext)s"

            with yt_dlp.YoutubeDL(current_opts) as ydl:
                ydl.download(urls_to_download)

            print("Batch finished. Resting to cool down...")
            time.sleep(random.randint(30, 60))
        else:
            print(f"All videos for {ytbr} are up to date.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube video download pipeline")
    parser.add_argument(
        "--input-dir",
        default="videos",
        help="Directory containing video metadata (default: videos)",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory to save audio downloads (default: downloads)",
    )
    parser.add_argument(
        "--youtubers",
        nargs="*",
        default=[],
        help="Filter to specific youtubers (default: all)",
    )
    args = parser.parse_args()

    video_ids = collect_video_ids(args.input_dir)

    if args.youtubers:
        video_ids = {k: v for k, v in video_ids.items() if k in args.youtubers}

    download_videos(video_ids, args.output_dir)


if __name__ == "__main__":
    main()
