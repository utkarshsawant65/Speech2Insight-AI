"""Task 1: Audio to text using OpenAI Whisper (report: Turbo for speed → base/small)."""

import tempfile
from pathlib import Path

import whisper

from .config import WHISPER_MODEL


def load_whisper_model(model_name: str = WHISPER_MODEL):
    """Load Whisper model (tiny, base, small, medium, large)."""
    return whisper.load_model(model_name)


def transcribe_audio(
    audio_path: str | Path,
    model=None,
    model_name: str = WHISPER_MODEL,
    language: str | None = None,
) -> str:
    """
    Transcribe audio file to text.
    Accepts mp3, wav, etc. (Whisper/ffmpeg handle conversion).
    """
    if model is None:
        model = load_whisper_model(model_name)
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    return (result.get("text") or "").strip()


def transcribe_uploaded_file(uploaded_file, model=None, model_name: str = WHISPER_MODEL) -> str:
    """Transcribe a Streamlit UploadedFile (save to temp then transcribe)."""
    suffix = Path(uploaded_file.name).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return transcribe_audio(tmp_path, model=model, model_name=model_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
