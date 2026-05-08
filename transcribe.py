"""Transcribe an audio file using Deepgram with speaker diarization."""

import json
import os
import subprocess
import sys
from pathlib import Path

from deepgram import DeepgramClient
from deepgram.core.request_options import RequestOptions
from dotenv import load_dotenv

load_dotenv()


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


def prepare_audio(input_path: Path) -> Path:
    """Ensure audio is Opus-encoded before transcription.

    Encodes to a .transcribe.ogg sidecar unless the input is already a
    pure Opus stream with no video track.  Reuses the sidecar when it is
    newer than the source.
    """
    info = get_media_info(input_path)
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    already_opus = bool(audio_streams) and all(
        s.get("codec_name") == "opus" for s in audio_streams
    )

    if already_opus and not has_video:
        return input_path

    output_path = input_path.with_suffix(".transcribe.ogg")

    if (
        output_path.exists()
        and output_path.stat().st_mtime > input_path.stat().st_mtime
    ):
        print(f"Reusing existing Opus audio: {output_path.name}")
        return output_path

    reason = "Video detected." if has_video else f"Non-Opus audio ({input_path.suffix})."
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


def transcribe(input_path: Path) -> None:
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

    print("Sending to Deepgram (nova-3, diarization enabled)...\n")

    client = DeepgramClient(api_key=api_key)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
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
    transcript_path.write_text("".join(output_list), encoding="utf-8")
    print(f"\n\nTranscript saved to: {transcript_path}")


def main() -> None:
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        # Auto-detect: find the most recently modified .m4a in the project dir
        project_dir = Path(__file__).parent
        m4a_files = sorted(
            project_dir.glob("*.m4a"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not m4a_files:
            print("No .m4a files found in the project directory.", file=sys.stderr)
            print("Usage: uv run transcribe.py [audio_file.m4a]", file=sys.stderr)
            sys.exit(1)
        audio_path = m4a_files[0]

    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    transcribe(audio_path)


if __name__ == "__main__":
    main()
