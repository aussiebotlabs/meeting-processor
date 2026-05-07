"""Tests for record_audio.py."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from record_audio import (
    NORMALIZE_BLOCK_SECONDS,
    TARGET_DBOV,
    TARGET_SR,
    MicRecorder,
    asl_p56,
    mix_audio,
)


# ---------------------------------------------------------------------------
# asl_p56 unit tests
# ---------------------------------------------------------------------------


def test_asl_p56_returns_none_for_silence():
    """A block of zeros must return None (no active speech detected)."""
    x = np.zeros(TARGET_SR * 5, dtype=np.float32)
    assert asl_p56(x, TARGET_SR) is None


def test_asl_p56_returns_none_for_empty():
    """An empty array must return None."""
    assert asl_p56(np.array([]), TARGET_SR) is None


def test_asl_p56_matches_known_rms():
    """A 1 kHz sine at amplitude 0.1 should give asl_rms ≈ 0.1/sqrt(2) ≈ 0.0707."""
    fs = TARGET_SR
    duration = 10  # seconds — long enough for the algorithm to converge
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    x = (0.1 * np.sin(2 * math.pi * 1000 * t)).astype(np.float32)
    asl_rms = asl_p56(x, fs)
    assert asl_rms is not None
    expected = 0.1 / math.sqrt(2)
    assert abs(asl_rms - expected) / expected < 0.05  # within 5 %


# ---------------------------------------------------------------------------
# mix_audio integration tests
# ---------------------------------------------------------------------------


def _write_sine(path: Path, freq: float, amplitude: float, duration: float, sr: int, channels: int = 1) -> None:
    """Write a pure sine wave WAV file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    if channels == 2:
        wave = np.stack([wave, wave], axis=1)
    sf.write(path, wave, sr, subtype="FLOAT")


def test_mix_audio_output_is_target_sr(tmp_path: Path):
    """Mixed output must be TARGET_SR, mono, and the right length."""
    sr = TARGET_SR
    dur = 3.0  # seconds
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    _write_sine(sys_path, 440, 0.3, dur, sr)
    _write_sine(mic_path, 880, 0.3, dur, sr)

    mix_audio(sys_path, mic_path, out_path)

    assert out_path.exists()
    mixed, mixed_sr = sf.read(out_path)
    assert mixed_sr == TARGET_SR
    assert mixed.ndim == 1
    # Length should be ~dur * TARGET_SR (allow ±1 % rounding from resample_poly)
    assert abs(len(mixed) - int(dur * TARGET_SR)) < int(0.01 * dur * TARGET_SR) + 1


def test_mix_audio_pads_different_lengths(tmp_path: Path):
    """Shorter source is zero-padded to match the longer one."""
    sr = TARGET_SR
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    _write_sine(sys_path, 440, 0.3, 5.0, sr)
    _write_sine(mic_path, 880, 0.3, 2.0, sr)  # shorter

    mix_audio(sys_path, mic_path, out_path)

    mixed, _ = sf.read(out_path)
    expected_len = int(5.0 * TARGET_SR)
    assert abs(len(mixed) - expected_len) < 500  # within 500 samples


def test_mix_audio_normalises_to_target_level(tmp_path: Path):
    """When only one source is non-silent, the mix should be near TARGET_DBOV."""
    sr = TARGET_SR
    dur = 20.0  # seconds — long enough for P.56 to converge
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    # System audio at a high amplitude, mic silent
    _write_sine(sys_path, 440, 0.5, dur, sr)
    sf.write(mic_path, np.zeros(int(sr * dur), dtype=np.float32), sr, subtype="FLOAT")

    mix_audio(sys_path, mic_path, out_path)

    mixed, _ = sf.read(out_path, dtype="float32")
    asl_rms = asl_p56(mixed, TARGET_SR)
    assert asl_rms is not None
    measured_dbov = 20.0 * math.log10(asl_rms)
    assert abs(measured_dbov - TARGET_DBOV) < 2.0  # within 2 dB


def test_mix_audio_spans_multiple_blocks(tmp_path: Path):
    """Audio longer than NORMALIZE_BLOCK_SECONDS must be handled across blocks."""
    sr = TARGET_SR
    dur = NORMALIZE_BLOCK_SECONDS * 2 + 10  # spans 3 blocks
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    _write_sine(sys_path, 440, 0.3, dur, sr)
    _write_sine(mic_path, 880, 0.3, dur, sr)

    mix_audio(sys_path, mic_path, out_path)

    mixed, mixed_sr = sf.read(out_path)
    assert mixed_sr == TARGET_SR
    expected_len = int(dur * TARGET_SR)
    assert abs(len(mixed) - expected_len) < 1000


def test_mix_audio_resamples_mismatched_inputs(tmp_path: Path):
    """48 kHz stereo system + 44.1 kHz mono mic → 16 kHz mono output."""
    dur = 3.0
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    _write_sine(sys_path, 440, 0.3, dur, 48_000, channels=2)
    _write_sine(mic_path, 880, 0.3, dur, 44_100, channels=1)

    mix_audio(sys_path, mic_path, out_path)

    _, mixed_sr = sf.read(out_path)
    assert mixed_sr == TARGET_SR

    info = sf.info(out_path)
    assert info.channels == 1
    expected_frames = int(dur * TARGET_SR)
    assert abs(info.frames - expected_frames) < int(0.01 * dur * TARGET_SR) + 1


def test_mix_audio_silent_block_passthrough(tmp_path: Path):
    """A fully silent input must not cause division-by-zero and output near-silence."""
    sr = TARGET_SR
    dur = 5.0
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    sf.write(sys_path, np.zeros(int(sr * dur), dtype=np.float32), sr, subtype="FLOAT")
    sf.write(mic_path, np.zeros(int(sr * dur), dtype=np.float32), sr, subtype="FLOAT")

    # Must not raise
    mix_audio(sys_path, mic_path, out_path)

    mixed, _ = sf.read(out_path, dtype="float32")
    assert np.max(np.abs(mixed)) < 1e-4


def test_mix_audio_missing_files(tmp_path: Path, capsys):
    """Test that missing files are handled gracefully."""
    mix_audio(tmp_path / "nonexistent.wav", tmp_path / "mic.wav", tmp_path / "out.wav")
    captured = capsys.readouterr()
    assert "Warning: One or both audio files missing" in captured.out


def test_mix_audio_overwrites_existing_output(tmp_path: Path):
    """mix_audio should overwrite an existing output file without error."""
    sr = TARGET_SR
    dur = 2.0
    sys_path = tmp_path / "sys.wav"
    mic_path = tmp_path / "mic.wav"
    out_path = tmp_path / "out.wav"

    _write_sine(sys_path, 440, 0.3, dur, sr)
    _write_sine(mic_path, 880, 0.3, dur, sr)

    mix_audio(sys_path, mic_path, out_path)
    mix_audio(sys_path, mic_path, out_path)  # second call must not raise

    assert out_path.exists()


# ---------------------------------------------------------------------------
# MicRecorder tests (unchanged from original)
# ---------------------------------------------------------------------------


@patch("sounddevice.InputStream")
@patch("soundfile.SoundFile")
def test_mic_recorder_starts_and_stops(mock_sf, mock_sd, tmp_path: Path):
    """Test that MicRecorder orchestrates sounddevice and soundfile."""
    import time

    out_path = tmp_path / "mic.wav"
    recorder = MicRecorder(out_path)

    recorder.start()
    time.sleep(0.1)
    assert recorder._thread.is_alive()

    recorder.stop()
    assert not recorder._thread.is_alive()

    mock_sd.assert_called_once()
    mock_sf.assert_called_once()


# ---------------------------------------------------------------------------
# main() orchestration tests (unchanged from original)
# ---------------------------------------------------------------------------


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
