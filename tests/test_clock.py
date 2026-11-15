"""The brain hears the current date and time during long sessions, and knows where it is."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker import launch_facts
from talker.voice_loop import with_clock

EDT = timezone(timedelta(hours=-4), "EDT")


def test_time_notes_come_on_the_first_turn_every_ten_minutes_and_at_midnight(monkeypatch):
    monkeypatch.setattr(launch_facts, "location", lambda lookup=True: "Bronxville, NY")
    now = {"t": datetime(2026, 10, 7, 23, 40, tzinfo=EDT)}
    notes, said = [], []
    reply = with_clock(lambda text: said.append(text) or iter(["ok"]), notes.append,
                       every_s=600, clock=lambda: now["t"])
    reply("hi")
    assert len(notes) == 1 and "October 7, 2026" in notes[0] and "23:40" in notes[0] and "Bronxville, NY" in notes[0]
    now["t"] += timedelta(minutes=5); reply("still here")
    assert len(notes) == 1                                  # five minutes on: no new note
    now["t"] += timedelta(minutes=6); reply("and now?")
    assert len(notes) == 2 and "23:51" in notes[1]
    now["t"] += timedelta(minutes=9, seconds=30); reply("past midnight")
    assert len(notes) == 3 and "October 8, 2026" in notes[2]   # the date changed inside ten minutes
    assert said == ["hi", "still here", "and now?", "past midnight"]


def test_location_prefers_the_saved_setting(monkeypatch, tmp_path):
    from talker import local_settings
    path = str(tmp_path / "settings.json")
    monkeypatch.setattr(local_settings, "SETTINGS_FILE", path)
    monkeypatch.setattr(local_settings, "load", lambda p=path: {"location": "Hilton Head, SC"})
    assert launch_facts.location(lookup=False) == "Hilton Head, SC"
    monkeypatch.setattr(local_settings, "load", lambda p=path: {})
    launch_facts._located.clear()
    assert launch_facts.location(lookup=False) == "an unknown location"


def test_location_placeholder_and_now_text(monkeypatch):
    monkeypatch.setattr(launch_facts, "location", lambda lookup=True: "Bronxville, NY")
    text = launch_facts.expand("You are in {location}. {{tool clock}} stays.")
    assert text == "You are in Bronxville, NY. {{tool clock}} stays."
    line = launch_facts.now_text(datetime(2026, 10, 7, 22, 5, tzinfo=EDT))
    assert line.startswith("It is now Wednesday, October 7, 2026 (2026-10-07) at 22:05 EDT") and "Bronxville, NY" in line
