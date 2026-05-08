"""Tests for transcribe.py media handling and preprocessing."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from transcribe import get_media_info, prepare_audio, transcribe


@pytest.fixture
def mock_path(tmp_path):
    path = tmp_path / "test.mp4"
    path.write_text("dummy content")
    return path


def test_get_media_info_success():
    mock_stdout = json.dumps({
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "100.0", "size": "1000"}
    })
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=mock_stdout, returncode=0)
        info = get_media_info(Path("test.mp4"))
        assert info["streams"][0]["codec_type"] == "video"
        assert info["streams"][1]["codec_name"] == "aac"
        assert mock_run.call_args[0][0][0] == "ffprobe"


def test_get_media_info_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        info = get_media_info(Path("test.mp4"))
        assert info == {}


def test_prepare_audio_passthrough_for_opus_audio(mock_path):
    # Already Opus with no video: should be returned unchanged
    with patch("transcribe.get_media_info") as mock_info:
        mock_info.return_value = {
            "streams": [{"codec_type": "audio", "codec_name": "opus"}]
        }
        result = prepare_audio(mock_path)
        assert result == mock_path


def test_prepare_audio_encodes_non_opus_audio(mock_path):
    # Plain audio (e.g. wav/pcm) with no video: must be encoded to Opus
    with patch("transcribe.get_media_info") as mock_info, \
         patch("subprocess.run") as mock_run:
        mock_info.return_value = {
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}]
        }

        result = prepare_audio(mock_path)

        expected_output = mock_path.with_suffix(".transcribe.ogg")
        assert result == expected_output
        assert mock_run.call_args[0][0][0] == "ffmpeg"
        assert str(mock_path) in mock_run.call_args[0][0]
        assert str(expected_output) in mock_run.call_args[0][0]


def test_prepare_audio_extracts_for_video(mock_path):
    # Video file (aac audio): must be encoded to Opus regardless of audio codec
    with patch("transcribe.get_media_info") as mock_info, \
         patch("subprocess.run") as mock_run:
        mock_info.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ]
        }

        result = prepare_audio(mock_path)

        expected_output = mock_path.with_suffix(".transcribe.ogg")
        assert result == expected_output
        assert mock_run.call_args[0][0][0] == "ffmpeg"
        assert str(mock_path) in mock_run.call_args[0][0]
        assert str(expected_output) in mock_run.call_args[0][0]


def test_prepare_audio_encodes_video_with_opus_audio(mock_path):
    # Video file even with Opus audio must still be encoded (to strip video)
    with patch("transcribe.get_media_info") as mock_info, \
         patch("subprocess.run") as mock_run:
        mock_info.return_value = {
            "streams": [
                {"codec_type": "video", "codec_name": "vp9"},
                {"codec_type": "audio", "codec_name": "opus"},
            ]
        }

        result = prepare_audio(mock_path)

        expected_output = mock_path.with_suffix(".transcribe.ogg")
        assert result == expected_output
        assert mock_run.call_args[0][0][0] == "ffmpeg"


def test_prepare_audio_reuses_existing_if_newer(mock_path):
    output_path = mock_path.with_suffix(".transcribe.ogg")
    output_path.write_text("existing audio")

    import os
    import time
    os.utime(output_path, (time.time() + 100, time.time() + 100))

    # Trigger via non-Opus audio (no video)
    with patch("transcribe.get_media_info") as mock_info, \
         patch("subprocess.run") as mock_run:
        mock_info.return_value = {
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}]
        }

        result = prepare_audio(mock_path)

        assert result == output_path
        mock_run.assert_not_called()


@patch("transcribe.DeepgramClient")
@patch("transcribe.prepare_audio")
@patch("os.getenv")
def test_transcribe_saves_to_original_path(mock_getenv, mock_prepare, mock_dg, mock_path):
    mock_getenv.return_value = "fake_key"
    
    # Mock prepare_audio to return a DIFFERENT path (simulating extraction)
    extracted_path = mock_path.with_suffix(".transcribe.ogg")
    extracted_path.write_text("extracted")
    mock_prepare.return_value = extracted_path
    
    # Mock Deepgram response
    mock_response = MagicMock()
    mock_response.results.utterances = [
        MagicMock(speaker=0, transcript="Hello world", start=0.0, end=1.0)
    ]
    mock_dg.return_value.listen.v1.media.transcribe_file.return_value = mock_response
    
    transcribe(mock_path)
    
    # Verify transcript saved to mock_path.txt, NOT extracted_path.txt
    transcript_path = mock_path.with_suffix(".txt")
    assert transcript_path.exists()
    assert "[Speaker 0]" in transcript_path.read_text()
    assert "Hello world" in transcript_path.read_text()
    
    # Verify extracted_path.txt does NOT exist
    assert not extracted_path.with_suffix(".txt").exists()
