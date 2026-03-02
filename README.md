# YOUTUBE VIDEO TRANSCRIPTOR

downloads youtube videos with [yt-dlp](https://github.com/yt-dlp/yt-dlp) and extracts video
transcripts with [whisperx](https://github.com/m-bain/whisperX) using speaker diarization 
(with fallback to normal whisper)

## USAGE

download audio from youtube (reads video IDs from `videos/`):

```
python download.py [--youtubers NAME ...]
```

transcribe downloaded audio (reads `.wav` files from `downloads/`):

```
python transcript.py [--no-speakers] [--youtubers NAME ...]
```

`--no-speakers` disables speaker diarization and uses plain whisper instead.  
`--youtubers` filters to one or more channels by folder name (default: all).


## TODO:

- [ ] download logic: multiple inputs (json, .txt, args)
