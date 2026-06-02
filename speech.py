from __future__ import annotations

import gc
from typing import Any, Dict, Optional, Tuple

import torch
import whisperx
from transformers import pipeline
from whisperx.diarize import DiarizationPipeline


class TranscriptionModelCache:
    """
    Model cache manager for transcription, alignment and diarization models.
    """

    def __init__(self, device: str, hf_token: str | None = None) -> None:
        """
        Initialize the cache with a torch device and optional HuggingFace token.

        params:
        - device: torch device to use for inference
        - hf_token: optional HuggingFace token for diarization
        """
        self.device = device
        self.hf_token = hf_token
        self.whisper_model = None
        self.align_models: Dict[str, Tuple[Any, Any]] = {}
        self.diarize_model = None
        self.whisper_pipeline = None

    def get_whisper_model(self, compute_type: str = "float16") -> Any:
        """
        Get or load the cached WhisperX model.

        params:
        - compute_type: compute type for the model

        returns:
        - whisper model instance
        """
        if self.whisper_model is None:
            comp_type = compute_type
            if self.device == "cpu" and comp_type == "float16":
                comp_type = "int8"
            self.whisper_model = whisperx.load_model(
                "large-v3-turbo", self.device, compute_type=comp_type
            )
        return self.whisper_model

    def get_align_model(self, language_code: str) -> Tuple[Any, Any]:
        """
        Get or load the cached alignment model for a given language.

        params:
        - language_code: language code to align

        returns:
        - tuple of alignment model and metadata
        """
        if language_code not in self.align_models:
            self.align_models[language_code] = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self.align_models[language_code]

    def get_diarize_model(self) -> Optional[DiarizationPipeline]:
        """
        Get or load the cached diarization pipeline.

        returns:
        - diarization pipeline instance
        """
        if self.diarize_model is None and self.hf_token:
            self.diarize_model = DiarizationPipeline(
                token=self.hf_token, device=self.device
            )
        return self.diarize_model

    def get_whisper_pipeline(self) -> Any:
        """
        Get or load the cached HuggingFace Whisper pipeline.

        returns:
        - transformers whisper pipeline instance
        """
        if self.whisper_pipeline is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                dtype=torch_dtype,
                device=self.device,
                model_kwargs={"attn_implementation": "sdpa"} if self.device == "cuda" else {},
                chunk_length_s=30,
                batch_size=24,
            )
        return self.whisper_pipeline

    def clear(self) -> None:
        """
        Clear all loaded models from cache and empty cuda cache.
        """
        self.whisper_model = None
        self.align_models.clear()
        self.diarize_model = None
        self.whisper_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def transcribe_old(
    audio_path: str,
    device: str,
    model_cache: TranscriptionModelCache | None = None,
) -> dict:
    """
    Run transcription with transformers Whisper (no diarization).

    params:
    - audio_path: path to the audio file
    - device: torch device to use for inference
    - model_cache: optional cache to reuse the loaded pipeline

    returns:
    - dict with transcription result
    """
    if model_cache is not None:
        pipe = model_cache.get_whisper_pipeline()
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            dtype=torch_dtype,
            device=device,
            model_kwargs={"attn_implementation": "sdpa"} if device == "cuda" else {},
            chunk_length_s=30,
            batch_size=24,
        )

    result = pipe(audio_path, return_timestamps=True, language="en")

    if model_cache is None:
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
    model_cache: TranscriptionModelCache | None = None,
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
    - model_cache: optional cache to reuse the loaded models

    returns:
    - dict with transcription and speaker segments
    """
    if model_cache is not None:
        model = model_cache.get_whisper_model(compute_type=compute_type)
    else:
        model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=batch_size)

    # align segments
    lang = result["language"]
    if model_cache is not None:
        model_a, metadata = model_cache.get_align_model(lang)
    else:
        model_a, metadata = whisperx.load_align_model(
            language_code=lang, device=device
        )

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if model_cache is None:
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

    # diarization
    if model_cache is not None:
        diarize_model = model_cache.get_diarize_model()
    else:
        diarize_model = DiarizationPipeline(token=hf_token, device=device)

    if diarize_model is not None:
        # garbage collect and empty cuda cache before diarization
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        try:
            diarize_segments = diarize_model(audio, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                print("  ⚠️ diarization failed on cuda due to oom. retrying on cpu (might take a while)...")
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    # load diarization pipeline on cpu to avoid cuda oom
                    cpu_diarize_model = DiarizationPipeline(token=hf_token, device="cpu")
                    diarize_segments = cpu_diarize_model(audio, max_speakers=max_speakers)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as cpu_e:
                    print(f"  ❌ diarization failed on cpu as well: {cpu_e}")
                    raise cpu_e
            else:
                raise e

    if model_cache is None:
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
