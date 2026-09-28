# Tests for audio_input.listener. No live mic/VAD-model calls: trim_silence is pure numpy logic,
# and the rest is checked via a signature smoke test.
import inspect

import numpy as np

from olympia.audio_input import listener


def test_trim_silence_trims_leading_and_trailing_silence():
    frame_size = 4
    # 5 frames: silence, silence, speech, speech, silence
    audio = np.arange(20, dtype=np.float32)
    speech_flags = [False, False, True, True, False]

    trimmed = listener.trim_silence(audio, speech_flags, frame_size)

    np.testing.assert_array_equal(trimmed, audio[8:16])


def test_trim_silence_no_speech_returns_empty():
    audio = np.arange(12, dtype=np.float32)
    speech_flags = [False, False, False]

    trimmed = listener.trim_silence(audio, speech_flags, 4)

    assert trimmed.size == 0


def test_trim_silence_speech_until_end():
    frame_size = 4
    audio = np.arange(12, dtype=np.float32)
    speech_flags = [False, True, True]

    trimmed = listener.trim_silence(audio, speech_flags, frame_size)

    np.testing.assert_array_equal(trimmed, audio[4:12])


def test_public_functions_exist_with_expected_signatures():
    # Smoke test: listen_for_wake_word/record_utterance/record_followup are tightly coupled to
    # live mic + model I/O, so we only verify shape here rather than mocking the whole audio stack.
    assert callable(listener.listen_for_wake_word)
    assert inspect.signature(listener.listen_for_wake_word).parameters == {}

    sig = inspect.signature(listener.record_utterance)
    assert list(sig.parameters) == ["timeout_s"]

    sig = inspect.signature(listener.record_followup)
    assert list(sig.parameters) == ["timeout_s"]
