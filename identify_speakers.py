"""Interactively identify speaker names in a diarized transcript using GPT-4.1-mini."""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT_HEADER = """\
You are helping identify the real names of speakers in a meeting transcript.
Speakers are labeled [Speaker 0], [Speaker 1], etc.

When asked about a speaker, respond ONLY with a JSON object with these fields:
  "name"           - your best guess at the real name, or null if unknown
  "confidence"     - "low", "medium", or "high"
  "evidence_quote" - a short verbatim quote from the transcript supporting your guess
  "reasoning"      - a brief explanation of why you think this is the person's name

If you have already been told the name of another speaker, use that context to help
identify remaining speakers (e.g. someone might address another speaker by name).

TRANSCRIPT:
"""

CONFIDENCE_COLOURS = {
    "high": "\033[92m",    # green
    "medium": "\033[93m",  # yellow
    "low": "\033[91m",     # red
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def discover_speakers(transcript: str) -> list[str]:
    """Return unique speaker labels in order of first appearance."""
    seen: dict[str, None] = {}
    for match in re.finditer(r"\[Speaker (\d+)\]", transcript):
        label = f"Speaker {match.group(1)}"
        seen[label] = None
    return list(seen.keys())


def apply_name_map(transcript: str, name_map: dict[str, str]) -> str:
    """Replace [Speaker N] labels using the confirmed name_map."""
    result = transcript
    for label, name in name_map.items():
        result = result.replace(f"[{label}]", f"[{name}]")
    return result


def parse_guess(content: str) -> dict:
    """Parse the model's JSON response, returning a safe default on failure."""
    try:
        data = json.loads(content)
        return {
            "name": data.get("name"),
            "confidence": data.get("confidence", "low"),
            "evidence_quote": data.get("evidence_quote", ""),
            "reasoning": data.get("reasoning", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"name": None, "confidence": "low", "evidence_quote": "", "reasoning": ""}


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def prompt_user(speaker: str, guess: dict) -> str | None:
    """
    Present the model's guess to the user and return the confirmed name.
    Returns None if the user skips this speaker.
    """
    conf_colour = CONFIDENCE_COLOURS.get(guess["confidence"], "")
    conf_label = _colour(guess["confidence"].upper(), conf_colour)

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{speaker}{RESET}")

    if guess["name"]:
        print(f"  Model guess : {BOLD}{guess['name']}{RESET} [{conf_label}]")
    else:
        print(f"  Model guess : {DIM}unknown{RESET} [{conf_label}]")

    if guess["reasoning"]:
        print(f"  Reasoning   : {guess['reasoning']}")

    if guess["evidence_quote"]:
        short_quote = guess["evidence_quote"][:200].replace("\n", " ")
        print(f"  Evidence    : {DIM}\"{short_quote}\"{RESET}")

    name_hint = f' "{guess["name"]}"' if guess["name"] else ""
    print()
    print(f"  [Enter] accept{name_hint}"
          "  |  type a name to override  |  [s] skip  |  [?] ask follow-up")

    while True:
        raw = input("  > ").strip()
        if raw == "":
            return guess["name"]
        if raw.lower() == "s":
            return None
        if raw == "?":
            return None  # handled by caller
        return raw


def ask_followup(speaker: str, question: str, messages: list[dict], client: OpenAI) -> str:
    """Send a free-form follow-up question about a speaker and print the response."""
    messages.append({"role": "user", "content": f"[Follow-up about {speaker}]: {question}"})
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    answer = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": answer})
    return answer


def identify_speakers(
    transcript_path: Path,
    client: OpenAI,
) -> dict[str, str]:
    """
    Interactively identify speakers in a transcript.
    Returns a mapping of 'Speaker N' -> confirmed name for confirmed speakers only.
    """
    transcript = transcript_path.read_text(encoding="utf-8")
    speakers = discover_speakers(transcript)

    if not speakers:
        print("No speakers found in transcript.")
        return {}

    print(f"\nFound {len(speakers)} speakers: {', '.join(speakers)}")
    print("Loading transcript into model context (this may take a moment)…")

    system_content = SYSTEM_PROMPT_HEADER + transcript
    messages: list[dict] = [{"role": "system", "content": system_content}]

    name_map: dict[str, str] = {}

    for speaker in speakers:
        # Ask the model for a guess
        messages.append({"role": "user", "content": f"Who is {speaker}? Reply as JSON."})
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw_content = resp.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": raw_content})

        guess = parse_guess(raw_content)

        # Interactive loop — allow follow-up questions
        while True:
            confirmed = prompt_user(speaker, guess)

            if confirmed == "?":
                question = input("  Your question: ").strip()
                if question:
                    answer = ask_followup(speaker, question, messages, client)
                    print(f"\n  {answer}\n")
                continue
            break

        if confirmed is not None:
            name_map[speaker] = confirmed
            messages.append({
                "role": "user",
                "content": f"Confirmed: {speaker} is '{confirmed}'. Remember this for the rest.",
            })
            messages.append({
                "role": "assistant",
                "content": f"Understood. {speaker} = '{confirmed}'.",
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Skipped: {speaker} name is unknown for now.",
            })
            messages.append({
                "role": "assistant",
                "content": f"Understood. {speaker} name remains unconfirmed.",
            })

    return name_map


def write_named_transcript(transcript_path: Path, name_map: dict[str, str]) -> Path:
    """Write a copy of the transcript with speaker labels replaced by names."""
    transcript = transcript_path.read_text(encoding="utf-8")
    renamed = apply_name_map(transcript, name_map)
    output_path = transcript_path.with_suffix(".named.txt")
    output_path.write_text(renamed, encoding="utf-8")
    return output_path


def print_summary(name_map: dict[str, str], all_speakers: list[str]) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}Speaker name summary{RESET}")
    print(f"{'─' * 60}")
    for spk in all_speakers:
        name = name_map.get(spk)
        if name:
            print(f"  {spk:<12} → {BOLD}{name}{RESET}")
        else:
            print(f"  {spk:<12} → {DIM}(skipped, kept as-is){RESET}")


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        transcript_path = Path(sys.argv[1])
    else:
        project_dir = Path(__file__).parent
        txt_files = sorted(
            [p for p in project_dir.glob("*.txt") if ".named" not in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not txt_files:
            print("No .txt transcript files found. Pass a file path as argument.", file=sys.stderr)
            sys.exit(1)
        transcript_path = txt_files[0]

    if not transcript_path.exists():
        print(f"File not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    print(f"{BOLD}Meeting Speaker Identifier{RESET}")
    print(f"Transcript: {transcript_path.name}")

    client = OpenAI(api_key=api_key)

    transcript = transcript_path.read_text(encoding="utf-8")
    all_speakers = discover_speakers(transcript)

    name_map = identify_speakers(transcript_path, client)

    print_summary(name_map, all_speakers)

    if name_map:
        output_path = write_named_transcript(transcript_path, name_map)
        print(f"\nNamed transcript written to: {BOLD}{output_path}{RESET}")
    else:
        print("\nNo names confirmed — output file not written.")


if __name__ == "__main__":
    main()
