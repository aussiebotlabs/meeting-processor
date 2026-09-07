# Meeting Processor

Records, transcribes, and diarizes meetings on macOS. Captures system audio and microphone simultaneously, mixes them into a clean mono file, sends it to Deepgram for speaker-diarized transcription, then uses GPT-4.1-mini to interactively identify speaker names.

## Pipeline

```
record_audio.py  →  recordings/<timestamp>.system.wav
                    recordings/<timestamp>.mic.wav
                    recordings/<timestamp>.mixed.wav   (P.56-normalized mix)

transcribe.py    →  <file>.txt                        (diarized transcript)

identify_speakers.py  →  <file>.named.txt             (names substituted)
```

## Prerequisites

- macOS (system audio capture requires [`catap`](https://github.com/nicholaswilde/catap), which uses a virtual audio driver)
- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `ffmpeg` / `ffprobe` in `PATH` (for video-to-audio extraction in `transcribe.py`)

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your API keys
```

**.env**

```
DEEPGRAM_KEY=<your Deepgram API key>
OPENAI_API_KEY=<your OpenAI API key>
```

## Usage

### 1. Record a meeting

```bash
uv run record_audio.py
```

You will be prompted to select a microphone. Recording continues until you press **Ctrl+C**. Three files are written to `recordings/`:

| File | Contents |
|------|----------|
| `<timestamp>.system.wav` | Raw system audio |
| `<timestamp>.mic.wav` | Raw microphone audio |
| `<timestamp>.mixed.wav` | P.56-normalized mono mix at 16 kHz |

Options:

```
--duration <seconds>   Stop automatically after N seconds
--mic <index|name>     Skip interactive microphone selection
--output <prefix>      Override the default timestamp-based filename prefix
```

### 2. Transcribe

```bash
uv run transcribe.py [audio_file]
```

Without an argument, the most recently modified `.m4a` in the project directory is used. Supports any audio or video file. Files with a video track, or with container bitrate above 128 kbps, are re-encoded to Opus before upload; lower-bitrate audio (e.g. voice-call M4A) is uploaded as-is.

Output: `<input>.txt` — a diarized transcript with `[Speaker N][MM:SS.ss]` headers.

### 3. Identify speakers

```bash
uv run identify_speakers.py [transcript.txt]
```

Without an argument, the most recently modified `.txt` (excluding `.named.txt` files) in the project directory is used.

GPT-4.1-mini reads the full transcript and guesses each speaker's name. For each speaker you can:

- **Enter** — accept the model's guess
- **Type a name** — override with your own
- **`s`** — skip (label is kept as `Speaker N`)
- **`?`** or **`? <question>`** — ask a follow-up question about the speaker

Output: `<transcript>.named.txt` with `[Speaker N]` labels replaced by confirmed names.

## Development

```bash
uv run pytest          # run all tests
uv run pytest -v       # verbose
```
