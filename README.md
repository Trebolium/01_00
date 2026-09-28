# Olympia — Conversational Voice AI

Always-listening voice assistant for the elderly. Listens for "Hey Olympia", transcribes and
analyzes speech for biometric cues (pace, pitch, loudness), asks an LLM for a response informed
by those cues, and speaks the answer back.

## Pipeline

1. **Wake word** — mic always listening for the wake phrase (`src/olympia/audio_input`)
2. **VAD + recording** — records the utterance, trims silence (700ms trailing silence = stop)
3. **ASR** — Groq Whisper turbo transcribes the audio (`src/olympia/asr`)
4. **Biometrics** — words-per-minute, average pitch, average loudness (`src/olympia/biometrics`)
5. **LLM** — OpenRouter (Gemini Flash 3.8) responds, using biometrics as context (`src/olympia/llm`)
6. **TTS** — edge-tts synthesizes and plays the reply (`src/olympia/tts`)
7. Loop: listen for the next turn; end the conversation after 5s of silence.

Each stage lives in its own module behind a small function-level interface (see
`src/olympia/types.py` for the shared data contracts), so any stage can be swapped for another
implementation without touching the rest of the pipeline.

## Setup

```bash
uv sync
cp .env.example .env   # fill in GROQ_API_KEY and OPENROUTER_API_KEY
uv run python -m olympia.orchestrator
```

## Tests

```bash
uv run pytest test/
```

## Docker

```bash
docker build -t olympia .
docker run --env-file .env olympia
```
