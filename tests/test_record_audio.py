"""Tests for record_audio.py."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from record_audio import MicRecorder, mix_audio


def test_mix_audio_pads_and_mixes(tmp_path: Path):
    """Test that mixing handles different lengths and mono conversion."""
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    # Create dummy audio files
    # System: 1 second of 440Hz sine wave (stereo)
    sr = 44100
    t = np.linspace(0, 1, sr)
    sys_data = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1)
    sf.write(sys_path, sys_data, sr)

    # Mic: 0.5 seconds of 880Hz sine wave (mono)
    t_mic = np.linspace(0, 0.5, sr // 2)
    mic_data = np.sin(2 * np.pi * 880 * t_mic)
    sf.write(mic_path, mic_data, sr)

    mix_audio(sys_path, mic_path, out_path)

    assert out_path.exists()
    mixed_data, mixed_sr = sf.read(out_path)
    assert mixed_sr == sr
    assert len(mixed_data) == len(sys_data)  # Padded to max length
    assert mixed_data.ndim == 1  # Mixed to mono


def test_mix_audio_values_are_half_sum(tmp_path: Path):
    """Verify the mixed signal equals (sys + mic) * 0.5 sample-for-sample."""
    sr = 44100
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    rng = np.random.default_rng(42)
    sys_data = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
    mic_data = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
    sf.write(sys_path, sys_data, sr, subtype="FLOAT")
    sf.write(mic_path, mic_data, sr, subtype="FLOAT")

    mix_audio(sys_path, mic_path, out_path)

    mixed, _ = sf.read(out_path, dtype="float32")
    expected = (sys_data + mic_data) * 0.5
    # WAV PCM_16 output has ~3e-5 per-sample quantisation error
    np.testing.assert_allclose(mixed, expected, atol=5e-5)


def test_mix_audio_spans_multiple_blocks(tmp_path: Path):
    """Ensure mixing works correctly when audio is longer than one block."""
    from record_audio import _MIX_BLOCK_SIZE

    sr = 44100
    # 3× block size so we exercise the loop properly
    n_frames = _MIX_BLOCK_SIZE * 3
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    sys_data = np.zeros(n_frames, dtype=np.float32)
    mic_data = np.ones(n_frames, dtype=np.float32) * 0.4
    sf.write(sys_path, sys_data, sr, subtype="FLOAT")
    sf.write(mic_path, mic_data, sr, subtype="FLOAT")

    mix_audio(sys_path, mic_path, out_path)

    mixed, _ = sf.read(out_path, dtype="float32")
    assert len(mixed) == n_frames
    # WAV PCM_16 output has ~3e-5 per-sample quantisation error
    np.testing.assert_allclose(mixed, 0.2, atol=5e-5)


def test_mix_audio_samplerate_mismatch_warns(tmp_path: Path, capsys):
    """A mismatch in sample rates should print a warning."""
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    sf.write(sys_path, np.zeros(100, dtype=np.float32), 44100)
    sf.write(mic_path, np.zeros(100, dtype=np.float32), 16000)

    mix_audio(sys_path, mic_path, out_path)

    captured = capsys.readouterr()
    assert "Sample rates differ" in captured.out


def test_mix_audio_missing_files(tmp_path: Path, capsys):
    """Test that missing files are handled gracefully."""
    mix_audio(tmp_path / "nonexistent.wav", tmp_path / "mic.wav", tmp_path / "out.wav")
    captured = capsys.readouterr()
    assert "Warning: One or both audio files missing" in captured.out


@patch("sounddevice.InputStream")
@patch("soundfile.SoundFile")
def test_mic_recorder_starts_and_stops(mock_sf, mock_sd, tmp_path: Path):
    """Test that MicRecorder orchestrates sounddevice and soundfile."""
    out_path = tmp_path / "mic.wav"
    recorder = MicRecorder(out_path)
    
    # Start in a thread
    recorder.start()
    
    # Give it a moment to start
    import time
    time.sleep(0.1)
    
    assert recorder._thread.is_alive()
    
    # Stop it
    recorder.stop()
    assert not recorder._thread.is_alive()
    
    mock_sd.assert_called_once()
    mock_sf.assert_called_once()


@patch("record_audio.record_system_audio")
@patch("record_audio.MicRecorder")
@patch("record_audio.mix_audio")
@patch("time.sleep", return_value=None)
def test_main_orchestration_duration(mock_sleep, mock_mix, mock_mic_recorder, mock_catap):
    """session.start and session.close are called on a normal timed run."""
    from record_audio import main

    mock_session = MagicMock()
    mock_catap.return_value = mock_session
    mock_mic_inst = MagicMock()
    mock_mic_recorder.return_value = mock_mic_inst

    with patch("sys.argv", ["record_audio.py", "--duration", "0.1", "--output", "test"]):
        with patch("pathlib.Path.mkdir"):
            main()

    mock_catap.assert_called_once()
    mock_session.start.assert_called_once()
    mock_mic_inst.start.assert_called_once()
    mock_session.close.assert_called_once()
    mock_mic_inst.stop.assert_called_once()
    mock_mix.assert_called_once()


@patch("record_audio.record_system_audio")
@patch("record_audio.MicRecorder")
@patch("record_audio.mix_audio")
@patch("time.sleep", side_effect=[None, KeyboardInterrupt])
def test_main_orchestration_keyboard_interrupt(mock_sleep, mock_mix, mock_mic_recorder, mock_catap):
    """session.close and mic_recorder.stop are called even when Ctrl+C is pressed."""
    from record_audio import main

    mock_session = MagicMock()
    mock_catap.return_value = mock_session
    mock_mic_inst = MagicMock()
    mock_mic_recorder.return_value = mock_mic_inst

    with patch("sys.argv", ["record_audio.py", "--duration", "10", "--output", "test"]):
        with patch("pathlib.Path.mkdir"):
            main()

    mock_session.start.assert_called_once()
    mock_mic_inst.start.assert_called_once()
    mock_session.close.assert_called_once()
    mock_mic_inst.stop.assert_called_once()
    mock_mix.assert_called_once()


@patch("record_audio.record_system_audio")
@patch("record_audio.MicRecorder")
@patch("record_audio.mix_audio")
@patch("time.sleep", side_effect=[None, KeyboardInterrupt])
def test_main_orchestration_indefinite_keyboard_interrupt(mock_sleep, mock_mix, mock_mic_recorder, mock_catap):
    """Without --duration, session.close and mic_recorder.stop are called on Ctrl+C."""
    from record_audio import main

    mock_session = MagicMock()
    mock_catap.return_value = mock_session
    mock_mic_inst = MagicMock()
    mock_mic_recorder.return_value = mock_mic_inst

    with patch("sys.argv", ["record_audio.py", "--output", "test"]):
        with patch("pathlib.Path.mkdir"):
            main()

    mock_session.start.assert_called_once()
    mock_session.close.assert_called_once()
    mock_mic_inst.stop.assert_called_once()
