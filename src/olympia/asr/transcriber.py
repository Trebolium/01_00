# ASR module — sends in-memory audio to Groq's Whisper endpoint for transcription.
import io
import logging

import numpy as np
import soundfile as sf
from groq import Groq

from olympia.config import ASR_MODEL, GROQ_API_KEY
from olympia.types import TranscriptResult, Word

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def transcribe(audio: np.ndarray, sample_rate: int) -> TranscriptResult:
    """Transcribe in-memory audio via Groq's whisper-large-v3-turbo, with word timestamps."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty. Set GROQ_API_KEY=<your key> in your .env file."
        )

    logger.info("Writing %d samples (%d Hz) to in-memory WAV buffer...", len(audio), sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    logger.info("Sending audio to Groq ASR model '%s'...", ASR_MODEL)
    client = Groq(api_key=GROQ_API_KEY)
    response = client.audio.transcriptions.create(
        file=("audio.wav", buffer.read()),
        model=ASR_MODEL,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

    text = response.text
    logger.info("Transcript received: %r", text)

    raw_words = getattr(response, "words", None) or []
    words = [_parse_word(w) for w in raw_words]
    logger.info("Parsed %d word-level timestamps.", len(words))

    return TranscriptResult(text=text, words=words)


def _parse_word(w) -> Word:
    # Groq may return words as dicts or as objects (pydantic extra fields) — handle both.
    get = w.get if isinstance(w, dict) else lambda k: getattr(w, k)
    return Word(text=get("word"), start=get("start"), end=get("end"))
