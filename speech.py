from __future__ import annotations

import gc

import torch
import whisperx
from transformers import pipeline
from whisperx.diarize import DiarizationPipeline


def transcribe_old(audio_path: str, device: str) -> dict:
    """
    Run transcription with transformers Whisper (no diarization).

    params:
    - audio_path: path to the audio file
    - device: torch device to use for inference

    returns:
    - dict with transcription result
    """
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
    max_speakers: int = 5,
    compute_type: str = "float16",
    batch_size: int = 16,
) -> dict:
    """
    Transcribe audio with WhisperX and speaker diarization.

    params:
    - audio_path: path to the audio file
    - hf_token: HuggingFace authentication token
    - device: torch device to use for inference
    - max_speakers: maximum number of speakers to detect
    - compute_type: compute type for the model
    - batch_size: batch size for transcription

    returns:
    - dict with transcription and speaker segments
    """
    model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    del model, diarize_model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def format_plain_text(result: dict) -> str:
    """
    Extract plain text from transcription result.

    params:
    - result: transcription result dict

    returns:
    - str with plain text
    """
    return result.get("text", "").strip()


def format_speakers_to_txt(result: dict) -> str:
    """
    Convert WhisperX result to [SPEAKER_X] text format.

    params:
    - result: transcription result with speaker information

    returns:
    - str with formatted speaker lines
    """
    lines = []
    for segment in result.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)
