import os
import json

import torch
import yt_dlp
from transformers import pipeline

import random
import time


def extract_transcriptions():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model on {device}...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
        chunk_length_s=30,
        batch_size=24,
    )

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
            print(f"transcribing audios {i}/{len(audios)} of youtuber {y}...")
            file_path = os.path.join(audio_dir, audio_file)
            _name = os.path.splitext(audio_file)[0]

            text_path = f"./text/{y}/{_name}.txt"
            if not os.path.exists(text_path):
                result = pipe(file_path,
                              return_timestamps=True,
                              language="en",
                              )

                with open(f"./text/{y}/{_name}.txt", "w", encoding="utf-8") as f:
                    f.write(result["text"])
            else:
                print(f"[SKIP] ./text/{y}/{_name} already exists.")



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
    # ids = collect_video_ids()
    # download_videos(ids)
    extract_transcriptions()


if __name__ == "__main__":
    main()
