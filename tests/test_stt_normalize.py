"""Quiet utterances are lifted to a fixed peak before the voice detector and Whisper."""
import numpy as np

from talker.stt_backends import normalize_level


def test_quiet_speech_is_lifted_silence_and_loud_are_left():
    quiet = np.sin(np.arange(1600) / 7.0).astype(np.float32) * 0.05
    out = normalize_level(quiet)
    assert abs(np.abs(out).max() - 0.7) < 1e-3
    loud = quiet * 16                                        # peak 0.8: already loud enough
    assert np.array_equal(normalize_level(loud), loud)
    hiss = quiet * 0.1                                       # peak 0.005: near silence, not amplified
    assert np.array_equal(normalize_level(hiss), hiss)
    assert normalize_level(np.zeros(0, dtype=np.float32)).size == 0
