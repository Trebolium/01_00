# Ties every module together into the full conversation loop. Run with: uv run python -m olympia.orchestrator
import logging

from olympia.asr.transcriber import transcribe
from olympia.audio_input.listener import (
    listen_for_wake_word,
    record_followup,
    record_utterance,
)
from olympia.biometrics.analyzer import analyze
from olympia.llm.responder import get_response
from olympia.tts.speaker import speak

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("olympia.orchestrator")


def _handle_turn(audio, sample_rate) -> None:
    """Run one utterance through ASR -> biometrics -> LLM -> TTS."""
    logger.info("Transcribing utterance...")
    transcript = transcribe(audio, sample_rate)

    logger.info("Analyzing voice biometrics...")
    features = analyze(audio, sample_rate, transcript)

    logger.info("Asking LLM for a response...")
    reply = get_response(transcript.text, features)

    logger.info("Speaking response...")
    speak(reply)


def run() -> None:
    logger.info("Olympia is starting up. Listening for the wake word...")
    while True:
        listen_for_wake_word()

        logger.info("Wake word detected — recording your message... (reverts to wake word if nothing is said within 5s)")
        result = record_utterance()
        if result is None:
            logger.info("No speech detected after wake word — back to listening for the wake word...")
            continue
        audio, sample_rate = result
        _handle_turn(audio, sample_rate)

        logger.info("Waiting for a follow-up... (conversation ends after 5s of silence)")
        while True:
            result = record_followup()
            if result is None:
                logger.info("No further speech detected — conversation ended, back to listening for the wake word...")
                break
            audio, sample_rate = result
            _handle_turn(audio, sample_rate)


if __name__ == "__main__":
    run()
