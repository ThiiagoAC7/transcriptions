# YOUTUBE VIDEO TRANSCRIPTOR

downloads youtube videos with [yt-dlp](https://github.com/yt-dlp/yt-dlp) and extracts video
transcripts with [whisperx](https://github.com/m-bain/whisperX) using speaker diarization 
(with fallback to normal whisper)

### TOPICS

attempt at understanding comment-video alignment. Treats video transcription and comments in one single
shared corpus, generates topics with BERTopic and computes alignment metrics with the topics probabilities
- warning: code is very weird
