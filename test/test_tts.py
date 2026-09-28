# Tests for speak() — mocks edge_tts synthesis and sounddevice playback (no network, no audio).
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from olympia.tts.speaker import DEFAULT_VOICE, speak


def test_speak_synthesizes_and_plays():
    fake_data = np.zeros(100, dtype=np.float32)
    fake_samplerate = 16000

    with patch("olympia.tts.speaker.edge_tts.Communicate") as mock_communicate_cls, \
         patch("olympia.tts.speaker.sf.read", return_value=(fake_data, fake_samplerate)) as mock_read, \
         patch("olympia.tts.speaker.sd.play") as mock_play, \
         patch("olympia.tts.speaker.sd.wait") as mock_wait, \
         patch("olympia.tts.speaker.os.remove") as mock_remove, \
         patch("olympia.tts.speaker.os.path.exists", return_value=True):
        mock_instance = MagicMock()
        mock_instance.save = AsyncMock()
        mock_communicate_cls.return_value = mock_instance

        speak("Hello there")

        mock_communicate_cls.assert_called_once_with("Hello there", DEFAULT_VOICE)
        mock_instance.save.assert_awaited_once()
        mock_read.assert_called_once()
        mock_play.assert_called_once_with(fake_data, fake_samplerate)
        mock_wait.assert_called_once()
        mock_remove.assert_called_once()


def test_speak_empty_text_returns_early():
    with patch("olympia.tts.speaker.edge_tts.Communicate") as mock_communicate_cls, \
         patch("olympia.tts.speaker.sd.play") as mock_play:
        speak("   ")

        mock_communicate_cls.assert_not_called()
        mock_play.assert_not_called()
