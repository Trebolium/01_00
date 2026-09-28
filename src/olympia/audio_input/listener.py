# Mic input: wake-word detection + VAD-gated utterance recording.
import logging
import time

import numpy as np
import sounddevice as sd

from olympia.config import CONVERSATION_SILENCE_TIMEOUT_S, SAMPLE_RATE, VAD_TRAILING_SILENCE_MS

logger = logging.getLogger("olympia.audio_input")

# openwakeword ships no "hey olympia" model, so we stand in with its pretrained "hey_jarvis" model.
WAKE_MODEL_NAME = "hey_jarvis"
WAKE_THRESHOLD = 0.5
OWW_FRAME_SAMPLES = 1280  # 80ms @ 16kHz, required frame size for openwakeword

VAD_FRAME_SAMPLES = 512  # required chunk size for silero-vad at 16kHz
VAD_SPEECH_THRESHOLD = 0.5
VAD_FRAME_MS = VAD_FRAME_SAMPLES / SAMPLE_RATE * 1000
VAD_TRAILING_SILENCE_FRAMES = int(VAD_TRAILING_SILENCE_MS / VAD_FRAME_MS)

_oww_model = None
_vad_model = None


def _get_oww_model():
    # Lazy singleton so the model loads once even if called repeatedly in a loop.
    global _oww_model
    if _oww_model is None:
        from openwakeword.model import Model

        logger.info("Loading openwakeword model '%s' (stand-in for 'hey olympia')...", WAKE_MODEL_NAME)
        logger.info("NOTE: first run downloads pretrained model files automatically.")
        # onnxruntime is the installed backend (tflite-runtime isn't available on all platforms, e.g. Apple Silicon).
        _oww_model = Model(wakeword_models=[WAKE_MODEL_NAME], inference_framework="onnx")
        logger.info("openwakeword model loaded.")
    return _oww_model


def _get_vad_model():
    # Lazy singleton for silero-vad; downloads via torch.hub on first use.
    global _vad_model
    if _vad_model is None:
        import torch

        logger.info("Loading silero-vad model via torch.hub (first run may download it)...")
        model, _utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        _vad_model = model
        logger.info("silero-vad model loaded.")
    return _vad_model


def _vad_speech_prob(chunk: np.ndarray) -> float:
    # Run silero-vad on a single 512-sample float32 chunk, return speech probability.
    import torch

    model = _get_vad_model()
    tensor = torch.from_numpy(chunk)
    with torch.no_grad():
        prob = model(tensor, SAMPLE_RATE).item()
    return prob


def trim_silence(audio: np.ndarray, speech_flags: list[bool], frame_size: int) -> np.ndarray:
    # Pure logic (testable without hardware): trim audio to the span between first and last speech frame.
    if not any(speech_flags):
        return np.array([], dtype=audio.dtype)
    first_idx = speech_flags.index(True)
    last_idx = len(speech_flags) - 1 - speech_flags[::-1].index(True)
    start_sample = first_idx * frame_size
    end_sample = min((last_idx + 1) * frame_size, len(audio))
    return audio[start_sample:end_sample]


def listen_for_wake_word() -> None:
    # Blocks until the wake phrase (stand-in "hey_jarvis") is detected on the live mic.
    logger.info("WAKE WORD STAND-IN: listening for 'hey_jarvis' as a placeholder for 'hey olympia'.")
    model = _get_oww_model()
    # Clear the model's internal audio-feature buffers so stale audio from a previous detection
    # (which we stop feeding as soon as it triggers) can't cause an immediate false re-trigger.
    model.reset()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=OWW_FRAME_SAMPLES) as stream:
        logger.info("Mic stream open. Waiting for wake word...")
        while True:
            frame, overflowed = stream.read(OWW_FRAME_SAMPLES)
            if overflowed:
                logger.warning("Input overflow detected, some audio may have been dropped.")
            frame = frame.reshape(-1)
            scores = model.predict(frame)
            score = scores.get(WAKE_MODEL_NAME, 0.0)
            if score >= WAKE_THRESHOLD:
                logger.info("Wake word detected! score=%.2f (stand-in for 'hey olympia')", score)
                return


def _record_with_vad(max_initial_wait_s: float | None) -> tuple[np.ndarray, int] | None:
    # Shared VAD-gated recording loop used by record_utterance and record_followup.
    frames: list[np.ndarray] = []
    speech_flags: list[bool] = []
    speech_started = False
    trailing_silence_frames = 0
    start_time = time.monotonic()

    logger.info("Recording started, waiting for speech...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=VAD_FRAME_SAMPLES) as stream:
        while True:
            chunk, overflowed = stream.read(VAD_FRAME_SAMPLES)
            if overflowed:
                logger.warning("Input overflow detected, some audio may have been dropped.")
            chunk = chunk.reshape(-1)
            frames.append(chunk)

            prob = _vad_speech_prob(chunk)
            is_speech = prob >= VAD_SPEECH_THRESHOLD
            speech_flags.append(is_speech)

            if is_speech:
                if not speech_started:
                    logger.info("Speech detected, recording utterance...")
                speech_started = True
                trailing_silence_frames = 0
            elif speech_started:
                trailing_silence_frames += 1
                if trailing_silence_frames >= VAD_TRAILING_SILENCE_FRAMES:
                    logger.info("%dms trailing silence reached, stopping recording.", VAD_TRAILING_SILENCE_MS)
                    break

            if not speech_started and max_initial_wait_s is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= max_initial_wait_s:
                    logger.info("No speech detected within %ss timeout, ending.", max_initial_wait_s)
                    return None

    full_audio = np.concatenate(frames)
    trimmed = trim_silence(full_audio, speech_flags, VAD_FRAME_SAMPLES)
    logger.info("Utterance recorded: %.2fs of trimmed audio.", len(trimmed) / SAMPLE_RATE)
    return trimmed, SAMPLE_RATE


def record_utterance(timeout_s: float = CONVERSATION_SILENCE_TIMEOUT_S) -> tuple[np.ndarray, int] | None:
    # Call right after wake word triggers. Records until VAD sees trailing silence, trims edges.
    # Returns None if no speech starts within timeout_s (caller should end the session).
    return _record_with_vad(max_initial_wait_s=timeout_s)


def record_followup(timeout_s: float = CONVERSATION_SILENCE_TIMEOUT_S) -> tuple[np.ndarray, int] | None:
    # For subsequent conversation turns (no wake word gate). Returns None if silence persists past timeout_s.
    return _record_with_vad(max_initial_wait_s=timeout_s)
