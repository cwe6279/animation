"""Piper: word spans from phoneme alignments, and the backend through the real pipeline."""
import os
import time
from types import SimpleNamespace as NS

import pytest

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from talker.tts_backends import piper_word_spans


def _al(*pairs):
    return [NS(phoneme=p, num_samples=n) for p, n in pairs]


def test_word_spans_split_on_spaces_and_skip_markers():
    al = _al(("^", 100), ("h", 50), ("i", 50), (" ", 20), ("ð", 30), ("ɛ", 40), (",", 60), (" ", 10),
             ("f", 20), ("ɹ", 30), (".", 40), ("$", 10))
    assert piper_word_spans(al) == [(100, 200), (220, 350), (360, 450)]
    assert piper_word_spans([]) == []


@pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.cache/talker/piper/en_US-hfc_female-medium.onnx")),
                    reason="Piper voice not downloaded")
def test_piper_backend_streams_words_and_audio():
    from talker.audio_engine import NullAudioEngine
    from talker.phoneme_scheduler import ScheduleReader
    from talker.speech_pipeline import SpeechPipeline
    from talker.tts_backends import PiperBackend
    from tests.test_pipeline import wait_until
    be = PiperBackend()
    p = SpeechPipeline(NullAudioEngine(sample_rate=be.sample_rate), ScheduleReader(), be, lead_seconds=0.0)
    p.start()
    try:
        t0 = time.monotonic()
        p.speak("Hello there, friend. I don't bite!")
        assert wait_until(lambda: p.stats is not None and p.stats.get("words"), timeout=15)
    finally:
        p.stop()
    assert p.stats["words"] == 6
    assert p.stats["audio_seconds"] > 1.0
    assert p.stats["time_to_first_audio_ms"] < 1500, p.stats
    print("piper first audio", p.stats["time_to_first_audio_ms"], "ms; total", round((time.monotonic() - t0) * 1000), "ms")
