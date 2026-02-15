"""Tests for transcribe module (mocked Whisper to avoid heavy deps in CI)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.transcribe import (
    check_ffmpeg_available,
    load_whisper_model,
    transcribe_audio,
    transcribe_uploaded_file,
)


def test_load_whisper_model_is_callable() -> None:
    """load_whisper_model is callable; real model load is integration (slow)."""
    assert callable(load_whisper_model)


@patch("src.transcribe.check_ffmpeg_available")
def test_transcribe_audio_uses_model(_mock_ffmpeg: MagicMock, tmp_path: Path) -> None:
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"text": "Hello world"}

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake wav content")

    out = transcribe_audio(audio_file, model=fake_model)
    assert out == "Hello world"
    fake_model.transcribe.assert_called_once()


@patch("src.transcribe.check_ffmpeg_available")
def test_transcribe_audio_missing_file(_mock_ffmpeg: MagicMock) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe_audio("/nonexistent/path/to/audio.mp3")


def test_check_ffmpeg_available_raises_when_missing() -> None:
    # Ensure auto-provision does not run (no imageio_ffmpeg) so which("ffmpeg") stays None
    with patch("src.transcribe._ensure_ffmpeg_on_path"):
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="ffmpeg"):
                check_ffmpeg_available()


def test_check_ffmpeg_available_passes_when_found() -> None:
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        check_ffmpeg_available()  # no raise


@patch("src.transcribe.check_ffmpeg_available")
@patch("src.transcribe.transcribe_audio")
def test_transcribe_uploaded_file_calls_transcribe_audio(
    mock_transcribe: MagicMock, _mock_ffmpeg: MagicMock
) -> None:
    mock_transcribe.return_value = "Transcribed text"
    fake_upload = MagicMock()
    fake_upload.name = "test.mp3"
    fake_upload.getvalue.return_value = b"fake audio bytes"

    out = transcribe_uploaded_file(fake_upload)
    assert out == "Transcribed text"
    mock_transcribe.assert_called_once()
