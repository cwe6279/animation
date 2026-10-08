"""Sleep hours: nothing goes to cloud speech recognition at night."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from talker import sleep_hours
from talker.sleep_hours import SleepHours, parse
from talker.voice_loop import VoiceLoop
from test_voice_loop import FakeSpeaker, ScriptedSTT


def at(h, m=0, day=8):
    return datetime(2026, 10, day, h, m)


def test_the_window_crosses_midnight_and_off_means_never():
    s = SleepHours("23:00-07:00")
    assert s.asleep(at(23, 30)) and s.asleep(at(2)) and s.asleep(at(6, 59))
    assert not s.asleep(at(7)) and not s.asleep(at(15)) and not s.asleep(at(22, 59))
    assert SleepHours("01:00-05:00").asleep(at(3)) and not SleepHours("01:00-05:00").asleep(at(6))
    assert not SleepHours("off").asleep(at(2)) and parse("") is None
    with pytest.raises(ValueError):
        parse("late")


def test_wake_now_lasts_until_the_window_ends_and_the_next_night_sleeps():
    s = SleepHours("23:00-07:00")
    s.wake_now(at(1))
    assert s.awake_until == at(7) and not s.asleep(at(3))
    assert not s.asleep(at(8)) and s.awake_until is None
    assert s.asleep(at(23, 30))                          # the next night sleeps as usual


class PausableSTT(ScriptedSTT):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls, self.fed = [], 0

    def pause(self): self.calls.append("pause")
    def resume(self): self.calls.append("resume")

    def feed(self, pcm):
        self.fed += 1
        return super().feed(pcm)


def test_asleep_she_sends_nothing_and_news_waits_for_the_morning(monkeypatch):
    now = {"t": at(23, 30)}

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return now["t"]
    import datetime as dt_mod
    monkeypatch.setattr(dt_mod, "datetime", FakeDT)

    stt, events = PausableSTT(), []
    loop = VoiceLoop(stt, lambda t: iter(["ok"]), FakeSpeaker(), wake_words=["clara"],
                     on_event=lambda k, s: events.append((k, s)))
    loop.sleep_hours = SleepHours("23:00-07:00")
    frame = b"\0" * 3200
    loop._process(frame)
    assert loop.asleep and stt.calls == ["pause"] and stt.fed == 0
    assert any(k == "mode" and "asleep until 07:00" in s for k, s in events)
    assert loop.announce("Goal g1 finished.") is False   # retried later, in the morning

    now["t"] = at(7, 1, day=9)
    loop._sleep_checked = 0.0
    loop._process(frame)
    assert not loop.asleep and stt.calls == ["pause", "resume"] and stt.fed >= 1
