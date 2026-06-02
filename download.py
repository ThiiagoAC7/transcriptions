from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import yt_dlp

VideoIdMap = Dict[str, List[str]]

def update_video_ids(text_dir: str, base_dir: str):
    """
    Traverse text_dir to see if any vid in base_dir/<youtuber>/selected_videos.json 
    were already collected.

    params:
    - text_dir: directory with transcribed video_ids
    - base_dir: directory with selected videos

    """
    if not os.path.exists(text_dir):
        print(f"Directory {text_dir} not found.")
        return

    for youtuber in os.listdir(text_dir):
        yt_text_dir = os.path.join(text_dir, youtuber)
        if not os.path.isdir(yt_text_dir):
            continue

        vids_collected = {
            os.path.splitext(f)[0]
            for f in os.listdir(yt_text_dir)
            if f.endswith(".txt")
        }

        selected_path = os.path.join(base_dir, youtuber, "selected_videos.json")
        if not os.path.exists(selected_path):
            continue

        with open(selected_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = False
        for entry in data.get("selected_videos", []):
            if entry["video_id"] in vids_collected and not entry["collected"]:
                entry["collected"] = True
                updated = True

        if updated:
            with open(selected_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"Updated {youtuber} collected status in {selected_path}")

def collect_video_ids(base_dir: str) -> VideoIdMap:
    """
    Traverse base_dir to find 'selected_videos.json' files
    and collect video IDs that have not been collected yet.

    params:
    - base_dir: base directory to search for metadata files

    returns:
    - dict mapping youtuber name to list of video IDs
    """
    video_ids: VideoIdMap = {}
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return video_ids

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "selected_videos.json":
                file_path = os.path.join(root, file)
                youtuber = os.path.basename(root)
                video_ids[youtuber] = []
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    selected = data.get("selected_videos", [])
                    new_vids = []
                    skipped = 0
                    for entry in selected:
                        if entry.get("collected"):
                            skipped += 1
                        else:
                            new_vids.append(entry["video_id"])
                    video_ids[youtuber].extend(new_vids)
                    print(f"Loaded {len(new_vids)} new videos from {youtuber} ({skipped} already collected)")

    return video_ids


def download_videos(video_ids: VideoIdMap, output_dir: str) -> None:
    """
    Download audio from specific video IDs using .wav for better quality.

    params:
    - video_ids: dict mapping youtuber to list of video IDs
    - output_dir: base directory to save downloaded audio files

    """
    if not video_ids:
        print("no video IDs provided.")
        return

    # todo: this needs to be fixed/better organized.
    # had some issues with js runtime, cookies, videos being empty, ...
    # the current config appears to be stable, but anything can happen
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
        "remote_components": {"ejs:github"},
        # "extractor_args": {
        #     "youtube": {
        #         "player_client": ["web", "android", "ios", "mweb"], # fallbacks if web fails
        #     }
        # },
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
            time.sleep(random.randint(5, 10))
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
    parser.add_argument(
        "--update-only",
        action="store_true",
        default=False,
        help="Update existing downloads (default: False)",
    )
    args = parser.parse_args()

    if args.update_only:
        update_video_ids("./text/", args.input_dir)
        return

    video_ids = collect_video_ids(args.input_dir)

    if args.youtubers:
        video_ids = {k: v for k, v in video_ids.items() if k in args.youtubers}

    download_videos(video_ids, args.output_dir)


if __name__ == "__main__":
    main()
