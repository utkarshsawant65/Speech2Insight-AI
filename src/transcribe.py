"""Task 1: Audio to text using OpenAI Whisper (report: Turbo for speed → base/small)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import WHISPER_MODEL

# Message when ffmpeg is missing (WinError 2 on Windows)
FFMPEG_REQUIRED_MSG = (
    "ffmpeg is required for audio decoding but was not found on your PATH. "
    "Install from https://ffmpeg.org/download.html — Windows: add ffmpeg/bin to PATH, "
    "or run: winget install ffmpeg  /  choco install ffmpeg"
)


def check_ffmpeg_available() -> None:
    """Raise a clear error if ffmpeg is not on PATH (fixes WinError 2 on Windows)."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(FFMPEG_REQUIRED_MSG)


def load_whisper_model(model_name: str = WHISPER_MODEL) -> Any:
    """Load Whisper model (tiny/base/small/medium/large). Lazy-import for fast startup."""
    import whisper  # noqa: PLC0415  # lazy to reduce initial load time
    return whisper.load_model(model_name)


def transcribe_audio(
    audio_path: str | Path,
    model: Any = None,
    model_name: str = WHISPER_MODEL,
    language: str | None = None,
) -> str:
    """
    Transcribe audio file to text.
    Accepts mp3, wav, etc. (Whisper/ffmpeg handle conversion).
    Validates ffmpeg and file before loading model.
    """
    check_ffmpeg_available()
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if model is None:
        model = load_whisper_model(model_name)
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    return (result.get("text") or "").strip()


def transcribe_uploaded_file(
    uploaded_file: Any, model: Any = None, model_name: str = WHISPER_MODEL
) -> str:
    """
    Transcribe a Streamlit UploadedFile (save to temp then transcribe).
    Uses a simple filename under system temp for Windows compatibility.
    """
    check_ffmpeg_available()
    name = getattr(uploaded_file, "name", "audio.mp3")
    suffix = Path(name).suffix or ".mp3"
    # Use gettempdir() + simple name so path is short and valid on Windows
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper_")
    try:
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())  # type: ignore[union-attr]
        return transcribe_audio(tmp_path, model=model, model_name=model_name)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
