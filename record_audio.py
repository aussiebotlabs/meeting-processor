"""Record system audio and microphone audio in parallel on macOS."""

import argparse
import datetime
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as sp_signal
import sounddevice as sd
import soundfile as sf
from catap import record_system_audio

TARGET_SR: int = 16_000          # output sample rate for mixed file
TARGET_DBOV: float = -26.0       # target active speech level (dBov)
NORMALIZE_BLOCK_SECONDS: int = 180   # normalisation block length in seconds

# P.56 Method B constants
_P56_T: float = 0.03   # envelope smoothing time constant (s)
_P56_H: float = 0.2    # hangover time (s)
_P56_M: float = 15.9   # margin between ASL and threshold (dB)
_P56_THRES_NO: int = 15


def asl_p56(x: np.ndarray, fs: int) -> float | None:
    """Return the active speech level (linear RMS) via ITU-T P.56 Method B.

    Returns None when no active speech is detected (silent block).
    x must be a 1-D float array normalised to [-1, 1].
    """
    eps = np.finfo(float).eps
    x = x.astype(np.float64).ravel()
    x_len = len(x)
    if x_len == 0:
        return None

    g = math.exp(-1.0 / (fs * _P56_T))
    I = int(math.ceil(fs * _P56_H))  # hangover length in samples

    # Thresholds in float amplitude domain: 2^(j-15) for j = 0..14
    c = 2.0 ** np.arange(-_P56_THRES_NO, 0, dtype=np.float64)  # c[0]=2^-15 … c[14]=2^-1

    # Two-stage IIR envelope follower
    x_abs = np.abs(x)
    b_filt = [1.0 - g, 0.0]
    a_filt = [1.0, -g]
    p = sp_signal.lfilter(b_filt, a_filt, x_abs)
    q = sp_signal.lfilter(b_filt, a_filt, p)

    # Activity counters and hangover counters per threshold
    a = np.full(_P56_THRES_NO, -1, dtype=np.int64)
    hang = np.full(_P56_THRES_NO, I, dtype=np.int64)

    for k in range(x_len):
        for j in range(_P56_THRES_NO):
            if q[k] >= c[j]:
                a[j] += 1
                hang[j] = 0
            elif hang[j] < I:
                a[j] += 1
                hang[j] += 1
            else:
                break  # thresholds are ordered: if this one fails, higher ones will too

    # No activity detected at lowest threshold
    if a[0] == -1:
        return None

    a = a + 2  # bias correction from reference implementation
    sq = float(x @ x)

    AdB1 = 10.0 * math.log10(sq / a[0] + eps)
    CdB1 = 20.0 * math.log10(c[0] + eps)
    if AdB1 - CdB1 < _P56_M:
        return None

    AdB = np.zeros(_P56_THRES_NO)
    CdB = np.zeros(_P56_THRES_NO)
    Delta = np.zeros(_P56_THRES_NO)
    AdB[0] = AdB1
    CdB[0] = CdB1
    Delta[0] = AdB1 - CdB1

    for j in range(1, _P56_THRES_NO):
        AdB[j] = 10.0 * math.log10(sq / (a[j] + eps) + eps)
        CdB[j] = 20.0 * math.log10(c[j] + eps)

    for j in range(1, _P56_THRES_NO):
        if a[j] != 0:
            Delta[j] = AdB[j] - CdB[j]
            if Delta[j] <= _P56_M:
                asl_ms_log, _ = _bin_interp(AdB[j], AdB[j - 1], CdB[j], CdB[j - 1], _P56_M, 0.5)
                return 10.0 ** (asl_ms_log / 20.0)

    return None


def _bin_interp(upcount: float, lwcount: float, upthr: float, lwthr: float, margin: float, tol: float) -> tuple[float, float]:
    """Bisection interpolation used by P.56 to find the active speech level."""
    if tol < 0:
        tol = -tol

    if abs(upcount - upthr - margin) < tol:
        return upcount, upthr
    if abs(lwcount - lwthr - margin) < tol:
        return lwcount, lwthr

    midcount = (upcount + lwcount) / 2.0
    midthr = (upthr + lwthr) / 2.0
    iterno = 1

    while True:
        diff = midcount - midthr - margin
        if abs(diff) <= tol:
            break
        iterno += 1
        if iterno > 20:
            tol *= 1.1
        if diff > tol:
            midcount = (upcount + midcount) / 2.0
            midthr = (upthr + midthr) / 2.0
        else:
            midcount = (midcount + lwcount) / 2.0
            midthr = (midthr + lwthr) / 2.0

    return midcount, midthr


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Collapse a block to mono if it has multiple channels."""
    return np.mean(block, axis=1) if block.ndim > 1 else block


def _resample(block: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float array from src_rate to dst_rate using polyphase filter."""
    if src_rate == dst_rate:
        return block
    g = math.gcd(src_rate, dst_rate)
    return sp_signal.resample_poly(block, dst_rate // g, src_rate // g).astype(np.float32)


def _p56_gain(block: np.ndarray, fs: int) -> float:
    """Compute gain to apply so the block's active level reaches TARGET_DBOV.

    Returns 1.0 for silent blocks (no active speech detected by P.56).
    """
    asl_rms = asl_p56(block, fs)
    if asl_rms is None or asl_rms == 0.0:
        return 1.0
    target_rms = 10.0 ** (TARGET_DBOV / 20.0)
    return target_rms / asl_rms


def mix_audio(system_path: Path, mic_path: Path, output_path: Path) -> None:
    """Mix system and microphone audio into a single 16 kHz mono file.

    Each NORMALIZE_BLOCK_SECONDS block of each source is independently
    normalised to TARGET_DBOV (ITU-T P.56 active speech level) before mixing.
    Silent blocks are passed through unchanged (gain = 1.0).
    """
    if not system_path.exists() or not mic_path.exists():
        print("Warning: One or both audio files missing. Skipping mix.")
        return

    print(f"Mixing {system_path.name} and {mic_path.name}...")

    with sf.SoundFile(system_path) as sys_f, sf.SoundFile(mic_path) as mic_f:
        sys_sr = sys_f.samplerate
        mic_sr = mic_f.samplerate

        # How many source frames to read per block so that each resampled
        # block is exactly NORMALIZE_BLOCK_SECONDS * TARGET_SR frames.
        sys_block_src = int(NORMALIZE_BLOCK_SECONDS * sys_sr)
        mic_block_src = int(NORMALIZE_BLOCK_SECONDS * mic_sr)

        with sf.SoundFile(
            output_path, mode="w", samplerate=TARGET_SR, channels=1, subtype="PCM_16"
        ) as out_f:
            while True:
                sys_raw = _to_mono(sys_f.read(sys_block_src, dtype="float32"))
                mic_raw = _to_mono(mic_f.read(mic_block_src, dtype="float32"))

                if len(sys_raw) == 0 and len(mic_raw) == 0:
                    break

                sys_block = _resample(sys_raw, sys_sr, TARGET_SR)
                mic_block = _resample(mic_raw, mic_sr, TARGET_SR)

                # Pad shorter block so both are the same length
                max_len = max(len(sys_block), len(mic_block))
                if len(sys_block) < max_len:
                    sys_block = np.pad(sys_block, (0, max_len - len(sys_block)))
                if len(mic_block) < max_len:
                    mic_block = np.pad(mic_block, (0, max_len - len(mic_block)))

                sys_gain = _p56_gain(sys_block, TARGET_SR)
                mic_gain = _p56_gain(mic_block, TARGET_SR)

                mixed = sys_block * sys_gain + mic_block * mic_gain
                out_f.write(np.clip(mixed, -1.0, 1.0))

    print(f"Mixed file saved to: {output_path}")


class MicRecorder:
    """Recorder for microphone audio using sounddevice."""

    def __init__(self, output_path: Path, samplerate: int = 44100, channels: int = 1):
        self.output_path = output_path
        self.samplerate = samplerate
        self.channels = channels
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _record(self):
        try:
            with sf.SoundFile(
                self.output_path, mode="x", samplerate=self.samplerate, channels=self.channels
            ) as file:
                with sd.InputStream(
                    samplerate=self.samplerate, channels=self.channels, callback=lambda data, frames, time, status: file.write(data)
                ):
                    self.stop_event.wait()
        except Exception as e:
            print(f"Microphone recording error: {e}")

    def start(self):
        self._thread = threading.Thread(target=self._record)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        if self._thread:
            self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record system and microphone audio.")
    parser.add_argument("--duration", type=float, help="Duration to record in seconds")
    parser.add_argument("--output", type=str, help="Output base name (prefix)")
    args = parser.parse_args()

    # Setup output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("recordings")
    output_dir.mkdir(exist_ok=True)

    base_name = args.output if args.output else timestamp
    system_path = output_dir / f"{base_name}.system.wav"
    mic_path = output_dir / f"{base_name}.mic.wav"
    mixed_path = output_dir / f"{base_name}.mixed.wav"

    print(f"Starting parallel recording (Base: {base_name})...")
    print("Press Ctrl+C to stop early.")

    system_session = record_system_audio(output_path=str(system_path))
    mic_recorder = MicRecorder(mic_path)

    start_time = time.time()
    try:
        system_session.start()
        mic_recorder.start()
        try:
            if args.duration:
                remaining = args.duration
                while remaining > 0:
                    sleep_time = min(remaining, 0.1)
                    time.sleep(sleep_time)
                    remaining -= sleep_time
            else:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
    finally:
        try:
            system_session.close()
        finally:
            mic_recorder.stop()

    actual_duration = time.time() - start_time
    print(f"Recording finished. Duration: {actual_duration:.2f}s")
    print(f"System audio: {system_path}")
    print(f"Microphone: {mic_path}")

    # Mix the files
    mix_audio(system_path, mic_path, mixed_path)


if __name__ == "__main__":
    main()
