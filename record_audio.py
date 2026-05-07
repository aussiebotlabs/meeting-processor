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


def mix_audio(system_path: Path, mic_path: Path, output_path: Path) -> None:
    """Mix system and microphone audio into a single file."""
    if not system_path.exists() or not mic_path.exists():
        print("Warning: One or both audio files missing. Skipping mix.")
        return

    print(f"Mixing {system_path.name} and {mic_path.name}...")
    
    # Load audio
    sys_data, sys_samplerate = sf.read(system_path)
    mic_data, mic_samplerate = sf.read(mic_path)

    # Ensure same sample rate (simple check, no resampling for now)
    if sys_samplerate != mic_samplerate:
        print(f"Warning: Sample rates differ ({sys_samplerate} vs {mic_samplerate}). Mix might be misaligned.")

    # Ensure same number of channels (convert to mono for mixing if needed)
    if sys_data.ndim > 1:
        sys_data = np.mean(sys_data, axis=1)
    if mic_data.ndim > 1:
        mic_data = np.mean(mic_data, axis=1)

    # Pad shorter track
    max_len = max(len(sys_data), len(mic_data))
    sys_padded = np.zeros(max_len)
    mic_padded = np.zeros(max_len)
    sys_padded[:len(sys_data)] = sys_data
    mic_padded[:len(mic_data)] = mic_data

    # Mix with conservative gain
    mixed = (sys_padded + mic_padded) * 0.5
    
    sf.write(output_path, mixed, sys_samplerate)
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
