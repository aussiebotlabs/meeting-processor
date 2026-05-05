"""Tests for identify_speakers.py pure helper functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from identify_speakers import (
    apply_name_map,
    discover_speakers,
    parse_guess,
    write_named_transcript,
)


SAMPLE_TRANSCRIPT = """\

[Speaker 0]
  [00:00.48 → 00:03.28] So maybe Marcel could give an overview.

[Speaker 1]
  [00:03.76 → 00:07.52] Sure, I'm Marcel and I run the operations.

[Speaker 2]
  [00:08.00 → 00:12.00] Great, and I'm Sarah, the account manager.

[Speaker 0]
  [00:12.50 → 00:16.00] Thanks Marcel, thanks Sarah.

[Speaker 3]
  [00:16.50 → 00:20.00] Hi everyone, I'm Jake from IT.
"""


# --- discover_speakers ---

def test_discover_speakers_returns_in_order():
    speakers = discover_speakers(SAMPLE_TRANSCRIPT)
    assert speakers == ["Speaker 0", "Speaker 1", "Speaker 2", "Speaker 3"]


def test_discover_speakers_no_duplicates():
    speakers = discover_speakers(SAMPLE_TRANSCRIPT)
    assert len(speakers) == len(set(speakers))


def test_discover_speakers_empty_transcript():
    assert discover_speakers("") == []


def test_discover_speakers_no_speakers():
    assert discover_speakers("Hello world, no speaker labels here.") == []


def test_discover_speakers_single():
    text = "[Speaker 0]\n  [00:00.00 → 00:01.00] Hello."
    assert discover_speakers(text) == ["Speaker 0"]


# --- apply_name_map ---

def test_apply_name_map_replaces_all_occurrences():
    name_map = {"Speaker 0": "Alice", "Speaker 1": "Bob"}
    result = apply_name_map(SAMPLE_TRANSCRIPT, name_map)
    assert "[Alice]" in result
    assert "[Bob]" in result
    assert "[Speaker 0]" not in result
    assert "[Speaker 1]" not in result


def test_apply_name_map_preserves_unconfirmed():
    name_map = {"Speaker 1": "Marcel"}
    result = apply_name_map(SAMPLE_TRANSCRIPT, name_map)
    assert "[Marcel]" in result
    assert "[Speaker 0]" in result
    assert "[Speaker 2]" in result
    assert "[Speaker 3]" in result


def test_apply_name_map_empty_map():
    result = apply_name_map(SAMPLE_TRANSCRIPT, {})
    assert result == SAMPLE_TRANSCRIPT


def test_apply_name_map_does_not_modify_original():
    name_map = {"Speaker 0": "Alice"}
    original = SAMPLE_TRANSCRIPT
    apply_name_map(SAMPLE_TRANSCRIPT, name_map)
    assert SAMPLE_TRANSCRIPT == original


# --- parse_guess ---

def test_parse_guess_valid_json():
    content = json.dumps({
        "name": "Marcel",
        "confidence": "high",
        "evidence_quote": "I'm Marcel",
        "reasoning": "Explicitly introduces himself.",
    })
    result = parse_guess(content)
    assert result["name"] == "Marcel"
    assert result["confidence"] == "high"
    assert result["evidence_quote"] == "I'm Marcel"
    assert "reasoning" in result


def test_parse_guess_null_name():
    content = json.dumps({
        "name": None,
        "confidence": "low",
        "evidence_quote": "",
        "reasoning": "No name found.",
    })
    result = parse_guess(content)
    assert result["name"] is None


def test_parse_guess_invalid_json_returns_safe_default():
    result = parse_guess("not valid json {{{")
    assert result["name"] is None
    assert result["confidence"] == "low"
    assert result["evidence_quote"] == ""


def test_parse_guess_missing_fields_uses_defaults():
    content = json.dumps({"name": "Alice"})
    result = parse_guess(content)
    assert result["name"] == "Alice"
    assert result["confidence"] == "low"
    assert result["evidence_quote"] == ""
    assert result["reasoning"] == ""


def test_parse_guess_empty_string():
    result = parse_guess("")
    assert result["name"] is None
    assert result["confidence"] == "low"


# --- write_named_transcript ---

def test_write_named_transcript(tmp_path: Path):
    transcript_file = tmp_path / "meeting.txt"
    transcript_file.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    name_map = {"Speaker 0": "Alice", "Speaker 1": "Marcel"}

    output = write_named_transcript(transcript_file, name_map)

    assert output == tmp_path / "meeting.named.txt"
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "[Alice]" in content
    assert "[Marcel]" in content
    assert "[Speaker 0]" not in content
    assert "[Speaker 1]" not in content
    # Original untouched
    assert transcript_file.read_text(encoding="utf-8") == SAMPLE_TRANSCRIPT


def test_write_named_transcript_partial_map(tmp_path: Path):
    transcript_file = tmp_path / "meeting.txt"
    transcript_file.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
    name_map = {"Speaker 1": "Marcel"}

    output = write_named_transcript(transcript_file, name_map)
    content = output.read_text(encoding="utf-8")

    assert "[Marcel]" in content
    assert "[Speaker 0]" in content  # unconfirmed, preserved
    assert "[Speaker 2]" in content


# --- identify_speakers integration (mocked OpenAI) ---

def _make_mock_completion(name: str, confidence: str = "high") -> MagicMock:
    """Build a fake chat completion response."""
    payload = json.dumps({
        "name": name,
        "confidence": confidence,
        "evidence_quote": f"I'm {name}",
        "reasoning": "Self-introduction.",
    })
    message = MagicMock()
    message.content = payload
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def test_identify_speakers_accepts_model_guess(tmp_path: Path):
    """When user just presses Enter, the model's guess is accepted."""
    from identify_speakers import identify_speakers

    transcript_file = tmp_path / "meeting.txt"
    transcript_file.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_mock_completion("Alice"),
        _make_mock_completion("Marcel"),
        _make_mock_completion("Sarah"),
        _make_mock_completion("Jake"),
    ]

    # Simulate user pressing Enter for all speakers
    with patch("builtins.input", return_value=""):
        result = identify_speakers(transcript_file, mock_client)

    assert result["Speaker 0"] == "Alice"
    assert result["Speaker 1"] == "Marcel"
    assert result["Speaker 2"] == "Sarah"
    assert result["Speaker 3"] == "Jake"


def test_identify_speakers_user_overrides_name(tmp_path: Path):
    """When user types a different name, that name is used instead."""
    from identify_speakers import identify_speakers

    transcript_file = tmp_path / "meeting.txt"
    transcript_file.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_mock_completion("WrongName"),
        _make_mock_completion("Marcel"),
        _make_mock_completion("Sarah"),
        _make_mock_completion("Jake"),
    ]

    # First speaker: user types "Alice". Rest: press Enter.
    with patch("builtins.input", side_effect=["Alice", "", "", ""]):
        result = identify_speakers(transcript_file, mock_client)

    assert result["Speaker 0"] == "Alice"
    assert result["Speaker 1"] == "Marcel"


def test_identify_speakers_skip_leaves_speaker_out(tmp_path: Path):
    """When user types 's', the speaker is excluded from the name map."""
    from identify_speakers import identify_speakers

    transcript_file = tmp_path / "meeting.txt"
    transcript_file.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_mock_completion("Alice"),
        _make_mock_completion("Marcel"),
        _make_mock_completion("Sarah"),
        _make_mock_completion("Jake"),
    ]

    # Skip Speaker 2; accept the rest
    with patch("builtins.input", side_effect=["", "", "s", ""]):
        result = identify_speakers(transcript_file, mock_client)

    assert "Speaker 2" not in result
    assert result["Speaker 0"] == "Alice"
    assert result["Speaker 3"] == "Jake"
