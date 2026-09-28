# Fast sanity tests for biometrics analyzer using short synthetic audio.
import numpy as np

from olympia.biometrics.analyzer import analyze
from olympia.types import TranscriptResult, Word

SAMPLE_RATE = 16000


def _sine_wave(freq_hz: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_words_per_minute_exact():
    audio = _sine_wave(150, 2.0)
    # 4 words spanning exactly 2.0s -> 4 words / (2/60 min) = 120 WPM
    words = [
        Word(text="one", start=0.0, end=0.4),
        Word(text="two", start=0.5, end=0.9),
        Word(text="three", start=1.0, end=1.4),
        Word(text="four", start=1.6, end=2.0),
    ]
    transcript = TranscriptResult(text="one two three four", words=words)

    result = analyze(audio, SAMPLE_RATE, transcript)

    assert result.words_per_minute == 120.0


def test_words_per_minute_empty_transcript_returns_zero():
    audio = _sine_wave(150, 1.0)
    transcript = TranscriptResult(text="", words=[])

    result = analyze(audio, SAMPLE_RATE, transcript)

    assert result.words_per_minute == 0.0


def test_pitch_estimation_ballpark():
    freq = 200.0
    audio = _sine_wave(freq, 1.5)
    words = [Word(text="hum", start=0.0, end=1.5)]
    transcript = TranscriptResult(text="hum", words=words)

    result = analyze(audio, SAMPLE_RATE, transcript)

    # pyin on a clean sine wave should land within ~10% of the true frequency
    assert abs(result.avg_pitch_hz - freq) / freq < 0.1


def test_loudness_is_finite_and_negative_for_quiet_audio():
    audio = _sine_wave(150, 1.0) * 0.01  # quiet signal
    words = [Word(text="hi", start=0.0, end=1.0)]
    transcript = TranscriptResult(text="hi", words=words)

    result = analyze(audio, SAMPLE_RATE, transcript)

    assert np.isfinite(result.avg_loudness_db)
    assert result.avg_loudness_db < 0
