# TTS module — synthesizes text via edge-tts and plays it back through default speakers.
import asyncio
import logging
import os
import tempfile

import edge_tts
import soundfile as sf
import sounddevice as sd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Swap this to change the voice (see `edge-tts --list-voices` for alternatives).
DEFAULT_VOICE = "en-US-AriaNeural"


def speak(text: str) -> None:
    """Synthesize `text` to speech and play it back, blocking until done."""
    if not text or not text.strip():
        logger.warning("speak() called with empty/whitespace-only text — skipping.")
        return

    tmp_path = tempfile.mktemp(suffix=".wav")
    try:
        logger.info("Starting TTS synthesis (voice=%s) for text: %r", DEFAULT_VOICE, text)
        asyncio.run(_synthesize(text, tmp_path))
        logger.info("TTS synthesis complete. Audio saved to %s", tmp_path)

        data, samplerate = sf.read(tmp_path)
        logger.info("Starting playback (%d Hz)...", samplerate)
        sd.play(data, samplerate)
        sd.wait()
        logger.info("Playback finished.")
    except Exception as e:
        raise RuntimeError(
            f"TTS synthesis/playback failed: {e}. "
            "Check your network connection (edge-tts requires internet) and that "
            "an audio output device is available."
        ) from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _synthesize(text: str, out_path: str) -> None:
    # edge-tts is async-only — this coroutine is invoked via asyncio.run() in speak().
    communicate = edge_tts.Communicate(text, DEFAULT_VOICE)
    await communicate.save(out_path)
