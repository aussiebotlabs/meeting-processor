"""Interactively identify speaker names in a diarized transcript using GPT-4.1-mini."""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

@dataclass
class PromptResult:
    action: Literal["accept", "override", "skip", "followup"]
    value: str | None = None

# Stable role description sent with every request (not task-specific).
# Task-specific instructions are added dynamically per call.
IDENTITY_SYSTEM_PROMPT = """\
You are helping identify the real names of speakers in a meeting transcript.
Speakers are labeled [Speaker 0], [Speaker 1], etc.

Reply ONLY with a JSON object with these fields:
  "name"           - your best guess at the real name, or null if unknown
  "confidence"     - "low", "medium", or "high"
  "evidence_quote" - a short verbatim quote from the transcript supporting your guess
  "reasoning"      - a brief explanation of why you think this is the person's name

If you have already been told the name of another speaker, use that context to help
identify remaining speakers (e.g. someone might address another speaker by name).
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are helping a user understand a meeting transcript.
Speakers are labeled [Speaker 0], [Speaker 1], etc.
Answer the question in concise plain English. Do NOT use JSON format.
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


def format_followup_answer(answer: str) -> str:
    """Render a follow-up answer as readable prose.

    If the model accidentally returns JSON (because it was primed by prior
    identity guesses), extract the most informative fields and return them as
    plain text rather than printing raw braces and keys.
    """
    stripped = answer.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            parts: list[str] = []
            if data.get("reasoning"):
                parts.append(data["reasoning"])
            if data.get("evidence_quote"):
                parts.append(f'Evidence: "{data["evidence_quote"]}"')
            if data.get("name"):
                confidence = data.get("confidence", "")
                label = f"{data['name']} ({confidence} confidence)" if confidence else data["name"]
                parts.append(f"Best guess: {label}")
            return "\n  ".join(parts) if parts else stripped
    except (json.JSONDecodeError, TypeError):
        pass
    return stripped


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def prompt_user(speaker: str, guess: dict) -> PromptResult:
    """Present the model's guess to the user and return the user's action."""
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
            if guess["name"]:
                return PromptResult("accept", guess["name"])
            return PromptResult("skip")
        if raw.lower() == "s":
            return PromptResult("skip")
        if raw == "?":
            return PromptResult("followup")
        if raw.startswith("?"):
            return PromptResult("followup", raw[1:].strip())
        return PromptResult("override", raw)


def ask_identity(
    speaker: str,
    transcript_message: dict,
    history: list[dict],
    client: OpenAI,
) -> dict:
    """Ask the model to identify a speaker, returning a parsed guess dict.

    The transcript message is always first (cached prefix), followed by
    confirmed-name history, then the task-specific system prompt and question.
    """
    messages = [
        transcript_message,
        *history,
        {"role": "system", "content": IDENTITY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Who is {speaker}?"},
    ]
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    return parse_guess(resp.choices[0].message.content or "{}")


def ask_followup(
    speaker: str,
    question: str,
    transcript_message: dict,
    history: list[dict],
    client: OpenAI,
) -> str:
    """Send a free-form follow-up question about a speaker, returning plain-text prose.

    Uses a separate system prompt that explicitly forbids JSON, so the model
    answers conversationally. Any accidentally-JSON response is reformatted by
    format_followup_answer before it reaches the user.
    """
    messages = [
        transcript_message,
        *history,
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        {"role": "user", "content": f"[Follow-up about {speaker}]: {question}"},
    ]
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    raw = resp.choices[0].message.content or ""
    return format_followup_answer(raw)


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

    # Stable prefix sent first in every request — long transcripts are cached here.
    transcript_message: dict = {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"}
    # Growing list of confirmation/skip messages accumulated across speakers.
    history: list[dict] = []

    name_map: dict[str, str] = {}

    for speaker in speakers:
        guess = ask_identity(speaker, transcript_message, history, client)

        # Interactive loop — allow follow-up questions before confirming
        confirmed_name = None
        while True:
            result = prompt_user(speaker, guess)

            if result.action == "followup":
                question = result.value
                if not question:
                    question = input("  Your question: ").strip()

                if question:
                    answer = ask_followup(speaker, question, transcript_message, history, client)
                    print(f"\n  {answer}\n")
                continue

            if result.action == "skip":
                confirmed_name = None
            else:
                confirmed_name = result.value
            break

        if confirmed_name is not None:
            name_map[speaker] = confirmed_name
            history.append({
                "role": "user",
                "content": f"Confirmed: {speaker} is '{confirmed_name}'. Remember this for the rest.",
            })
            history.append({
                "role": "assistant",
                "content": f"Understood. {speaker} = '{confirmed_name}'.",
            })
        else:
            history.append({
                "role": "user",
                "content": f"Skipped: {speaker} name is unknown for now.",
            })
            history.append({
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
