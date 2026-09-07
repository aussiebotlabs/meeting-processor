"""Transcribe an audio file using Deepgram with speaker diarization."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from deepgram import DeepgramClient
from deepgram.core.request_options import RequestOptions
from dotenv import load_dotenv

load_dotenv()

ENCODE_BITRATE_THRESHOLD_BPS = 128_000


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_media_info(path: Path) -> dict:
    """Get media information using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return {}


def container_bitrate_bps(info: dict, file_size: int) -> float | None:
    """Compute container bitrate from file size and duration."""
    duration_str = info.get("format", {}).get("duration")
    if duration_str is None:
        return None
    try:
        duration = float(duration_str)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return (file_size * 8) / duration


def should_encode_to_opus(info: dict, file_size: int) -> tuple[bool, str]:
    """Return whether to encode and a human-readable reason."""
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    if has_video:
        return True, "Video detected."

    bitrate = container_bitrate_bps(info, file_size)
    if bitrate is None:
        return True, "Unknown duration."

    if bitrate > ENCODE_BITRATE_THRESHOLD_BPS:
        kbps = int(round(bitrate / 1000))
        return True, f"High bitrate ({kbps} kbps)."

    return False, ""


def prepare_audio(input_path: Path) -> Path:
    """Ensure audio is suitable for transcription upload.

    Encodes to a .transcribe.ogg sidecar when the file has a video track or
    container bitrate exceeds ENCODE_BITRATE_THRESHOLD_BPS. Reuses the sidecar
    when it is newer than the source.
    """
    info = get_media_info(input_path)
    file_size = input_path.stat().st_size
    encode, reason = should_encode_to_opus(info, file_size)
    if not encode:
        return input_path

    output_path = input_path.with_suffix(".transcribe.ogg")

    if (
        output_path.exists()
        and output_path.stat().st_mtime > input_path.stat().st_mtime
    ):
        print(f"Reusing existing Opus audio: {output_path.name}")
        return output_path

    print(f"{reason} Encoding to Opus: {output_path.name}...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",  # No video
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16kHz
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"Warning: Failed to encode audio with ffmpeg ({e}). Uploading original file instead.",
            file=sys.stderr,
        )
        return input_path


def transcribe(input_path: Path, language: str = "en") -> None:
    api_key = os.getenv("DEEPGRAM_KEY")
    if not api_key:
        print("Error: DEEPGRAM_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    audio_path = prepare_audio(input_path)

    print(f"Transcribing: {input_path.name}")
    if audio_path != input_path:
        print(f"Using processed audio: {audio_path.name}")

    print(f"Original size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
    if audio_path != input_path:
        print(f"Upload size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")

    model = "nova-3"

    print(f"Sending to Deepgram ({model}, language={language}, diarization enabled)...\n")

    client = DeepgramClient(api_key=api_key)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model=model,
        language=language,
        diarize=True,
        smart_format=True,
        punctuate=True,
        utterances=True,
        paragraphs=True,
        filler_words=True,
        request_options=RequestOptions(timeout_in_seconds=900),
    )

    utterances = (
        response.results.utterances
        if response.results and response.results.utterances
        else []
    )

    if not utterances:
        print("No utterances found in the response.")
        if response.results and response.results.channels:
            channel = response.results.channels[0]
            if channel.alternatives:
                print(channel.alternatives[0].transcript or "")
        return

    output_list = []
    current_speaker = None

    for utterance in utterances:
        speaker = utterance.speaker
        transcript = utterance.transcript or ""
        start = utterance.start or 0.0
        # end = utterance.end or 0.0

        if speaker != current_speaker:
            current_speaker = speaker
            label = f"Speaker {speaker}" if speaker is not None else "Unknown"
            output_list.append(f"\n[{label}][{format_timestamp(start)}]\n  ")

        output_list.append(transcript)

    # Save transcript to file
    transcript_path = input_path.with_suffix(".txt")
    transcript_path.write_text(" ".join(output_list), encoding="utf-8")
    print(f"\n\nTranscript saved to: {transcript_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio using Deepgram.")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file")
    parser.add_argument(
        "--language",
        "-l",
        default="en",
        help="BCP-47 language code (e.g. en, de, fr, es). Default: en",
    )
    args = parser.parse_args()

    if args.audio_file:
        audio_path = Path(args.audio_file)
    else:
        # Auto-detect: find the most recently modified .m4a in the project dir
        project_dir = Path(__file__).parent
        m4a_files = sorted(
            project_dir.glob("*.m4a"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not m4a_files:
            print("No .m4a files found in the project directory.", file=sys.stderr)
            print("Usage: uv run transcribe.py [audio_file] [--language LANG]", file=sys.stderr)
            sys.exit(1)
        audio_path = m4a_files[0]

    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    transcribe(audio_path, language=args.language)


if __name__ == "__main__":
    main()
