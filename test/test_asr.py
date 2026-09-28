# Tests for ASR transcriber — mocks Groq API, no real network calls.
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

SAMPLE_RATE = 16000


def _sine_wave(freq_hz: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_transcribe_parses_mocked_response():
    from olympia.asr import transcriber

    mock_response = MagicMock()
    mock_response.text = "hello world"
    mock_response.words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
    ]

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch("olympia.asr.transcriber.GROQ_API_KEY", "fake-key"), patch(
        "olympia.asr.transcriber.Groq", return_value=mock_client
    ):
        result = transcriber.transcribe(_sine_wave(150, 1.0), SAMPLE_RATE)

    assert result.text == "hello world"
    assert len(result.words) == 2
    assert result.words[0].text == "hello"
    assert result.words[0].start == 0.0
    assert result.words[0].end == 0.4
    assert result.words[1].text == "world"
    assert result.words[1].end == 0.9

    # Verify the call requested word-level timestamps on the configured model.
    _, kwargs = mock_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == transcriber.ASR_MODEL
    assert kwargs["timestamp_granularities"] == ["word"]
    assert kwargs["response_format"] == "verbose_json"


def test_transcribe_raises_clear_error_when_api_key_missing():
    from olympia.asr import transcriber

    with patch("olympia.asr.transcriber.GROQ_API_KEY", ""):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            transcriber.transcribe(_sine_wave(150, 1.0), SAMPLE_RATE)
