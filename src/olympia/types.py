# Shared data contracts between modules — keep these stable so components stay swappable.
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class TranscriptResult:
    text: str
    words: list[Word]


@dataclass
class BiometricFeatures:
    words_per_minute: float
    avg_pitch_hz: float
    avg_loudness_db: float
