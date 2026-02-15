"""Task 1: Audio to text using OpenAI Whisper (report: Turbo for speed → base/small)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import WHISPER_MODEL

# Fallback message when ffmpeg cannot be provided (no system, no bundle)
FFMPEG_REQUIRED_MSG = (
    "ffmpeg is required for audio decoding. Install it: "
    "pip install imageio-ffmpeg  (bundled), or install ffmpeg and add to PATH."
)

_ffmpeg_path_ensured = False


def _ffmpeg_cache_dir() -> Path:
    """Platform-specific cache dir for bundled ffmpeg (industry standard: app data)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "nlp_audio_txt" / "ffmpeg_bin"


def _ensure_ffmpeg_on_path() -> None:
    """
    Ensure 'ffmpeg' is on PATH: use system ffmpeg if present, else bundle from
    imageio-ffmpeg (pip install) so users need no manual ffmpeg install.
    """
    global _ffmpeg_path_ensured  # noqa: PLW0603
    if _ffmpeg_path_ensured:
        return
    if shutil.which("ffmpeg") is not None:
        _ffmpeg_path_ensured = True
        return
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
    except ImportError:
        _ffmpeg_path_ensured = True
        return
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe or not os.path.isfile(exe):
        _ffmpeg_path_ensured = True
        return
    cache_dir = _ffmpeg_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    dest = cache_dir / ffmpeg_name
    try:
        if not dest.is_file() or os.path.getmtime(exe) > os.path.getmtime(dest):
            shutil.copy2(exe, dest)
    except OSError:
        _ffmpeg_path_ensured = True
        return
    path_sep = os.pathsep
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(cache_dir) + path_sep + old_path
    _ffmpeg_path_ensured = True


def check_ffmpeg_available() -> None:
    """
    Ensure ffmpeg is available (system or bundled via imageio-ffmpeg);
    raise with clear message only if neither is present.
    """
    _ensure_ffmpeg_on_path()
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
