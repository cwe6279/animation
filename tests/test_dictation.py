"""Dictation mode, spontaneous announcements and the session-end hook in VoiceLoop."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker.voice_loop import VoiceLoop
from test_voice_loop import FakeSpeaker, ScriptedSTT, wait


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def make(clock, events, replies=None):
    stt, spk = ScriptedSTT(), FakeSpeaker()
    heard = []
    def llm(text):
        heard.append(text)
        return iter(replies or ["ok."])
    loop = VoiceLoop(stt, llm, spk, clock=clock, on_event=lambda k, s: events.append((k, s)))
    loop.dictation_pause_s = 4.0
    loop.dictation_words = ["take this down"]
    loop.dictation_end_words = ["that's all"]
    return loop, stt, spk, heard


def test_dictation_holds_phrases_and_answers_them_together_after_the_pause():
    clock, events = Clock(), []
    loop, stt, spk, heard = make(clock, events)
    said = []
    loop.on_dictation = said.append
    loop.on_user_text("take this down")
    assert loop.dictating and said == [True]
    loop.on_user_text("First point, the budget is fine.")
    clock.t += 2
    loop.on_user_text("Second, we ship Thursday.")
    clock.t += 2
    loop.on_user_text("Third, tell Sam.")
    loop.tick()
    assert heard == [] and spk.spoken == []                 # 2 s gaps: still dictating
    assert [e for e in events if e[0] == "dictation"][-1][1].endswith("Third, tell Sam.")
    clock.t += 3.9
    loop.tick()
    assert heard == []
    clock.t += 0.2
    loop.tick()
    assert wait(lambda: len(spk.spoken) == 1)
    assert heard == ["First point, the budget is fine. Second, we ship Thursday. Third, tell Sam."]
    assert loop.dictating                                   # the mode stays on for the next paragraph
    loop.on_user_text("that's all")
    assert not loop.dictating and said == [True, False]


def test_end_word_answers_at_once_and_partial_speech_delays_the_flush():
    clock, events = Clock(), []
    loop, stt, spk, heard = make(clock, events)
    loop.set_dictation(True, "test")
    loop.on_user_text("Draft the memo.")
    clock.t += 10
    loop._partial = "and also"                              # someone is mid-sentence: wait
    loop.tick()
    assert heard == []
    loop._partial = ""
    loop.on_user_text("Keep it short.")
    loop.on_user_text("that's all")
    assert wait(lambda: len(spk.spoken) == 1)
    assert heard == ["Draft the memo. Keep it short."]
    assert not loop.dictating
    # not in dictation: a normal turn goes straight through
    loop.on_user_text("hello")
    assert wait(lambda: len(spk.spoken) == 2)
    assert heard[-1] == "hello"


def test_announce_only_when_the_room_is_quiet_and_lands_as_an_event():
    clock, events = Clock(), []
    loop, stt, spk, heard = make(clock, events)
    loop._last_busy = 0.0
    spk.busy = True
    assert loop.announce("task done") is False               # she is talking
    spk.busy = False
    loop._last_heard = time.monotonic()
    assert loop.announce("task done") is False               # someone just spoke
    loop._last_heard = 0.0
    loop._partial = "um"
    assert loop.announce("task done") is False               # someone is mid-sentence
    loop._partial = ""
    loop.dictating = True
    assert loop.announce("task done") is False               # never during dictation
    loop.dictating = False
    loop.set_waiting(True)
    assert loop.announce("task done") is False
    loop.set_waiting(False)
    assert loop.announce('Task 3f2a "find projectors" finished. Result: the Epson.') is True
    assert wait(lambda: len(spk.spoken) == 1)
    assert heard == ['(Event: Task 3f2a "find projectors" finished. Result: the Epson.)']
    assert ("event", 'Task 3f2a "find projectors" finished. Result: the Epson.') in events
    assert not any(k == "you" for k, _ in events)            # not passed off as the visitor


def test_session_end_fires_once_after_two_turns():
    clock, events = Clock(), []
    loop, stt, spk, heard = make(clock, events)
    ends = []
    loop.on_session_end = ends.append
    loop.end_session("early")
    assert ends == []                                        # nothing happened yet
    loop.on_user_text("one")
    assert wait(lambda: len(spk.spoken) == 1)
    loop.end_session("one turn")
    assert ends == []                                        # too short to be worth a note
    loop.on_user_text("two"); assert wait(lambda: len(spk.spoken) == 2)
    loop.on_user_text("three"); assert wait(lambda: len(spk.spoken) == 3)
    loop.end_session("window closed")
    loop.end_session("window closed")
    assert ends == ["window closed"]
