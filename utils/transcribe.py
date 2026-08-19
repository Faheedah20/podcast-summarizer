"""
Audio transcription utilities.
Supports local faster-whisper (recommended for development)
and OpenAI / Groq Whisper APIs (better for free Streamlit Cloud deployment).
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

# Windows may not support Hugging Face's symlink cache optimization.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st

from utils.config import load_project_env

load_project_env(Path(__file__).resolve().parent.parent)


@st.cache_resource(show_spinner="Loading the transcription model...")
def _load_local_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "0")),
        num_workers=1,
    )


def transcribe_local(audio_path: str, model_size: str = "tiny") -> str:
    """
    Transcribe using faster-whisper (runs locally, no API cost).
    model_size options: tiny, base, small, medium, large-v3
    """
    model = _load_local_model(model_size)

    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        language="en",
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    transcript_parts = []
    for segment in segments:
        start = _format_timestamp(segment.start)
        end = _format_timestamp(segment.end)
        text = segment.text.strip()
        if text:
            transcript_parts.append(f"[{start} - {end}] {text}")

    return "\n".join(transcript_parts).strip()


def transcribe_openai(audio_path: str) -> str:
    """Transcribe using OpenAI Whisper API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return _format_api_segments(transcript)


def transcribe_groq(audio_path: str) -> str:
    """Transcribe using Groq Whisper API (very fast)."""
    from groq import Groq

    max_file_size = 25 * 1024 * 1024
    file_size = os.path.getsize(audio_path)
    if file_size > max_file_size:
        raise ValueError(
            "Groq accepts audio files up to 25 MB. Compress or shorten this recording, "
            "or select Local transcription."
        )

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
        timeout=300.0,
        max_retries=3,
    )

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language="en"
        )
    return _format_api_segments(transcription)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as an easy-to-scan HH:MM:SS timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_api_segments(transcription) -> str:
    """Convert verbose API transcription segments into timestamped text."""
    if isinstance(transcription, dict):
        segments = transcription.get("segments", []) or []
    else:
        segments = getattr(transcription, "segments", None) or []
    formatted = []
    for segment in segments:
        if isinstance(segment, dict):
            text = segment.get("text", "").strip()
            start_value = segment.get("start", 0)
            end_value = segment.get("end", 0)
        else:
            text = getattr(segment, "text", "").strip()
            start_value = getattr(segment, "start", 0)
            end_value = getattr(segment, "end", 0)
        if text:
            start = _format_timestamp(start_value)
            end = _format_timestamp(end_value)
            formatted.append(f"[{start} - {end}] {text}")

    if formatted:
        return "\n".join(formatted)
    text = transcription.get("text", "") if isinstance(transcription, dict) else getattr(transcription, "text", str(transcription))
    return text.strip()


def transcribe_audio(audio_path: str, method: Optional[str] = None) -> str:
    """
    Main entry point.
    method: "local" | "openai" | "groq"
    Falls back to environment variable or defaults to local.
    """
    method = method or os.getenv("TRANSCRIPTION_METHOD", "local").lower()

    if method == "openai":
        if not _is_configured_key("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for OpenAI transcription")
        return transcribe_openai(audio_path)

    if method == "groq":
        if not _is_configured_key("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is missing or still set to the placeholder in .env")
        return transcribe_groq(audio_path)

    # Default: local faster-whisper
    model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
    return transcribe_local(audio_path, model_size=model_size)


def _is_configured_key(name: str) -> bool:
    """Return false for missing or copied example credentials."""
    value = os.getenv(name, "").strip()
    return bool(value) and not value.startswith("your_")


def save_uploaded_file(uploaded_file) -> str:
    """Save Streamlit uploaded file to a temporary location and return the path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name