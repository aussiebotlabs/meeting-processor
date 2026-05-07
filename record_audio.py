"""Record system audio and microphone audio in parallel on macOS."""

import argparse
import datetime
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from catap import record_system_audio

_MIX_BLOCK_SIZE = 4096  # frames per iteration (~93 ms at 44100 Hz)


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Collapse a block to mono if it has multiple channels."""
    return np.mean(block, axis=1) if block.ndim > 1 else block


def mix_audio(system_path: Path, mic_path: Path, output_path: Path) -> None:
    """Mix system and microphone audio into a single file, block by block."""
    if not system_path.exists() or not mic_path.exists():
        print("Warning: One or both audio files missing. Skipping mix.")
        return

    print(f"Mixing {system_path.name} and {mic_path.name}...")

    with sf.SoundFile(system_path) as sys_f, sf.SoundFile(mic_path) as mic_f:
        if sys_f.samplerate != mic_f.samplerate:
            print(
                f"Warning: Sample rates differ ({sys_f.samplerate} vs {mic_f.samplerate}). Mix might be misaligned."
            )

        with sf.SoundFile(
            output_path, mode="x", samplerate=sys_f.samplerate, channels=1
        ) as out_f:
            while True:
                sys_block = _to_mono(sys_f.read(_MIX_BLOCK_SIZE, dtype="float32"))
                mic_block = _to_mono(mic_f.read(_MIX_BLOCK_SIZE, dtype="float32"))

                if len(sys_block) == 0 and len(mic_block) == 0:
                    break

                max_len = max(len(sys_block), len(mic_block))
                if len(sys_block) < max_len:
                    sys_block = np.pad(sys_block, (0, max_len - len(sys_block)))
                if len(mic_block) < max_len:
                    mic_block = np.pad(mic_block, (0, max_len - len(mic_block)))

                out_f.write((sys_block + mic_block) * 0.5)

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
