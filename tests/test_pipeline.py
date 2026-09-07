import sys, os, asyncio, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from talker.audio_engine import NullAudioEngine
from talker.phoneme_scheduler import Emotion, ScheduleReader, Viseme
from talker.speech_pipeline import SpeechPipeline
from talker.tts_backends import (AudioChunk, END_OF_TEXT, ElevenLabsBackend, SentenceDone,
                          TTSBackend, WordBoundary)


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


class FakeBackend(TTSBackend):
    """Emits 0.2 s of audio per word and word boundaries 0.2 s apart."""
    name = "fake"
    sample_rate = 1000

    def __init__(self, delay=0.0):
        self.delay = delay
        self.seen = []

    async def synthesize(self, sentences):
        t = 0.0
        while True:
            s = await sentences.get()
            if s is END_OF_TEXT:
                return
            self.seen.append(s)
            await asyncio.sleep(self.delay)
            words = s.split()
            for w in words:
                yield WordBoundary(w, t, t + 0.2)
                t += 0.2
            # 0.2 s per word at 1 kHz int16 = 400 bytes per word
            yield AudioChunk(b"\x01\x00" * (200 * len(words)))
            yield SentenceDone(s)


def wait_until(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_null_engine_timeline():
    clk = FakeClock()
    eng = NullAudioEngine(sample_rate=1000, clock=clk)
    assert eng.enqueue_pcm(b"\x00" * 2000) == pytest.approx(0.0)   # 1 s of audio
    assert eng.enqueue_pcm(b"\x00" * 2000) == pytest.approx(1.0)   # queued right after
    clk.t = 5.0                                                     # idle gap
    assert eng.enqueue_pcm(b"\x00" * 2000) == pytest.approx(5.0)   # starts "now"
    assert eng.queued_seconds() == pytest.approx(1.0)
    eng.flush()
    assert eng.queued_seconds() == 0.0


def test_pipeline_places_events_on_timeline_with_emotions():
    clk = FakeClock()
    eng = NullAudioEngine(sample_rate=1000, clock=clk)
    sched = ScheduleReader()
    be = FakeBackend()
    p = SpeechPipeline(eng, sched, be, lead_seconds=0.0)
    p.start()
    try:
        clk.t = 3.0
        p.speak("[happy]one two three. [angry]four five.")
        assert wait_until(lambda: len(sched.visemes) > 0 and not p._active_sessions)
        assert be.seen == ["one two three.", "four five."]
        # first word starts where the session's audio was placed: t=3.0
        assert sched.visemes[0].time == pytest.approx(3.0)
        assert sched.end_time == pytest.approx(4.0)         # 5 words x 0.2 s
        assert sched.current_emotion(3.1) is Emotion.HAPPY
        assert sched.current_emotion(3.7) is Emotion.ANGRY  # fourth word at 3.6
        assert p.speech_end_time == pytest.approx(4.0)
        assert sched.current_viseme(3.05) is not Viseme.SIL
    finally:
        p.stop()


def test_pipeline_sessions_queue_back_to_back():
    clk = FakeClock()
    eng = NullAudioEngine(sample_rate=1000, clock=clk)
    sched = ScheduleReader()
    p = SpeechPipeline(eng, sched, FakeBackend(), lead_seconds=0.0)
    p.start()
    try:
        p.speak("alpha beta gamma delta")
        p.speak("epsilon zeta")
        assert wait_until(lambda: not p.is_busy or (not p._active_sessions and sched.end_time > 1.0))
        assert sched.end_time == pytest.approx(1.2)    # 6 words contiguous
    finally:
        p.stop()


def test_pipeline_interrupt_clears():
    clk = FakeClock()
    eng = NullAudioEngine(sample_rate=1000, clock=clk)
    sched = ScheduleReader()
    p = SpeechPipeline(eng, sched, FakeBackend(delay=0.3), lead_seconds=0.0)
    p.start()
    try:
        p.speak("slow sentence here. and another one here.")
        time.sleep(0.05)
        p.interrupt()
        time.sleep(0.5)
        assert sched.visemes == []
        assert not p.is_busy
    finally:
        p.stop()


def test_pipeline_stream_feeds_sentences_incrementally():
    clk = FakeClock()
    eng = NullAudioEngine(sample_rate=1000, clock=clk)
    sched = ScheduleReader()
    be = FakeBackend()
    p = SpeechPipeline(eng, sched, be, lead_seconds=0.0)
    p.start()
    try:
        def tokens():
            for t in ["Hello ", "there my ", "friend. ", "Second ", "sentence now."]:
                yield t
                time.sleep(0.02)
        p.speak_stream(tokens())
        assert wait_until(lambda: not p._active_sessions)
        assert be.seen == ["Hello there my friend.", "Second sentence now."]
    finally:
        p.stop()


def test_pipeline_reports_backend_error():
    class Broken(TTSBackend):
        async def synthesize(self, sentences):
            await sentences.get()
            raise RuntimeError("boom")
            yield
    errors = []
    p = SpeechPipeline(NullAudioEngine(), ScheduleReader(), Broken(), on_error=errors.append)
    p.start()
    try:
        p.speak("anything")
        assert wait_until(lambda: errors)
        assert "boom" in errors[0]
        assert not p.is_busy
    finally:
        p.stop()


def test_elevenlabs_alignment_to_words():
    al = {"chars": list("hi yo"), "charStartTimesMs": [0, 50, 100, 150, 200],
          "charDurationsMs": [50, 50, 50, 50, 50]}
    words = ElevenLabsBackend._alignment_words(al, base_t=1.0)
    assert [(w.word, round(w.start, 2), round(w.end, 2)) for w in words] == \
        [("hi", 1.0, 1.1), ("yo", 1.15, 1.25)]


def test_elevenlabs_wordizer_spans_chunks_and_skips_tags():
    from talker.tts_backends import _AlignmentWordizer
    w = _AlignmentWordizer(base_t=0.0)
    # "[sigh] hel" | "lo there" split across two HTTP chunks; tag straddles nothing here,
    # but the word "hello" does.
    c1 = {"characters": list("[si"), "character_start_times_seconds": [0, 0, 0], "character_end_times_seconds": [0, 0, 0]}
    c2 = {"characters": list("gh] hel"), "character_start_times_seconds": [0, 0, 0, 0, 1.0, 1.1, 1.2],
          "character_end_times_seconds": [0, 0, 0, 0, 1.1, 1.2, 1.3]}
    c3 = {"characters": list("lo there"), "character_start_times_seconds": [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
          "character_end_times_seconds": [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]}
    words = w.feed(c1) + w.feed(c2) + w.feed(c3) + w.flush()
    assert [(x.word, round(x.start, 1), round(x.end, 1)) for x in words] == [("hello", 1.0, 1.5), ("there", 1.6, 2.1)]
