# Tests for get_response() — mocks the OpenAI client (no real network calls).
from unittest.mock import MagicMock, patch

import pytest

from olympia.config import LLM_MODEL, LLM_SYSTEM_PROMPT
from olympia.llm.responder import get_response
from olympia.types import BiometricFeatures


def _fake_features():
    return BiometricFeatures(words_per_minute=120.0, avg_pitch_hz=180.5, avg_loudness_db=-20.0)


def test_get_response_builds_messages_and_returns_text():
    fake_reply = "I'm here to help, how are you feeling today?"

    with patch("olympia.llm.responder.OPENROUTER_API_KEY", "fake-key"), \
         patch("olympia.llm.responder.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=fake_reply))]
        mock_client.chat.completions.create.return_value = mock_response

        result = get_response("I feel a bit tired today", _fake_features())

        assert result == fake_reply
        mock_openai_cls.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="fake-key"
        )
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == LLM_MODEL
        messages = kwargs["messages"]
        assert messages[0] == {"role": "system", "content": LLM_SYSTEM_PROMPT}
        assert messages[1]["role"] == "user"
        assert "words-per-minute=120.0" in messages[1]["content"]
        assert "average pitch=180.5hz" in messages[1]["content"]
        assert "average loudness=-20.0db" in messages[1]["content"]
        assert "I feel a bit tired today" in messages[1]["content"]


def test_get_response_missing_api_key_raises():
    with patch("olympia.llm.responder.OPENROUTER_API_KEY", ""):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            get_response("hello", _fake_features())
