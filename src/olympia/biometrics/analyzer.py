# Extracts simple voice biometrics (pace, pitch, loudness) from an utterance's audio + transcript.
import logging

import librosa
import numpy as np

from olympia.types import BiometricFeatures, TranscriptResult

logger = logging.getLogger(__name__)

# Audio contract: mono float32/float64 numpy array, roughly normalized to [-1, 1] (see audio_input module).
EPS = 1e-10


def _words_per_minute(transcript: TranscriptResult) -> float:
    # WPM from first word start to last word end; guards divide-by-zero on tiny/degenerate spans.
    words = transcript.words
    if not words:
        logger.warning("No words in transcript — words_per_minute set to 0.0")
        return 0.0

    duration_s = words[-1].end - words[0].start
    duration_min = duration_s / 60.0
    if duration_min < 1e-6:
        logger.warning(
            "Transcript duration too short (%.4fs) to compute WPM — falling back to 0.0", duration_s
        )
        return 0.0

    wpm = len(words) / duration_min
    logger.info("words_per_minute = %.2f (%d words over %.2fs)", wpm, len(words), duration_s)
    return wpm


def _avg_pitch_hz(audio: np.ndarray, sample_rate: int) -> float:
    # librosa.pyin estimates f0 per frame; average only voiced frames.
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
    except Exception as e:
        logger.warning("librosa.pyin failed (%s) — avg_pitch_hz set to 0.0", e)
        return 0.0

    voiced_f0 = f0[voiced_flag]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    if voiced_f0.size == 0:
        logger.warning("No voiced frames detected — avg_pitch_hz set to 0.0")
        return 0.0

    pitch = float(np.mean(voiced_f0))
    logger.info("avg_pitch_hz = %.2f (%d/%d voiced frames)", pitch, voiced_f0.size, f0.size)
    return pitch


def _avg_loudness_db(audio: np.ndarray) -> float:
    # RMS energy per frame -> dB, averaged across frames. EPS guards log(0).
    rms = librosa.feature.rms(y=audio)[0]
    db = 20 * np.log10(rms + EPS)
    loudness = float(np.mean(db))
    logger.info("avg_loudness_db = %.2f", loudness)
    return loudness


def analyze(audio: np.ndarray, sample_rate: int, transcript: TranscriptResult) -> BiometricFeatures:
    # Runs all three biometric extractors and packages the result. Swap any sub-function to change approach.
    logger.info("Analyzing biometrics for utterance: %d samples @ %dHz, %d words", len(audio), sample_rate, len(transcript.words))

    wpm = _words_per_minute(transcript)
    pitch = _avg_pitch_hz(audio, sample_rate)
    loudness = _avg_loudness_db(audio)

    features = BiometricFeatures(
        words_per_minute=wpm,
        avg_pitch_hz=pitch,
        avg_loudness_db=loudness,
    )
    logger.info("Biometrics extracted: %s", features)
    return features
