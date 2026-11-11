"""Placeholders filled into a character's prompt at launch."""
from datetime import datetime, timedelta, timezone

from talker.launch_facts import builtin_facts, expand, register

NOON = datetime(2026, 9, 10, 14, 5, tzinfo=timezone(timedelta(hours=-4), "EDT"))


def test_builtin_placeholders_read_naturally():
    f = builtin_facts(NOON)
    assert f["date"] == "Thursday, September 10, 2026" and f["iso_date"] == "2026-09-10"
    assert f["time"] == "14:05" and f["weekday"] == "Thursday" and f["year"] == "2026"
    assert f["timezone"] == "EDT" and f["utc_offset"] == "UTC-04:00"


def test_expand_fills_known_and_leaves_everything_else():
    text = "It is {date} at {time} {timezone}. You are at {venue}. Nod with {{move nod}}. {unknown} stays."
    out = expand(text, {"venue": "the north gate"}, now=NOON)
    assert "It is Thursday, September 10, 2026 at 14:05 EDT." in out
    assert "You are at the north gate." in out
    assert "{{move nod}}" in out          # action blocks are not placeholders
    assert "{unknown} stays." in out      # unknown ones are left as written
    assert expand("", now=NOON) == "" and expand("no braces", now=NOON) == "no braces"


def test_a_face_can_override_and_code_can_register():
    register("weather", lambda: "clear and cold")
    out = expand("{weather} at {time}", {"time": "half past three"}, now=NOON)
    assert out == "clear and cold at half past three"     # the face's own value wins


def test_a_broken_registered_fact_does_not_break_the_prompt():
    def boom():
        raise RuntimeError("no network")
    register("forecast", boom)
    assert expand("today: {forecast}", now=NOON) == "today: {forecast}"


def test_clock_note_and_per_turn_wrapper():
    from datetime import datetime
    from talker.launch_facts import clock_note
    from talker.voice_loop import with_clock
    note = clock_note(datetime(2026, 11, 11, 8, 25))
    assert note.startswith("the clock says Wednesday, November 11, 2026 (2026-11-11) at 08:25")
    notes, turns = [], []
    wrapped = with_clock(lambda text: turns.append(text) or iter([text]), notes.append)
    list(wrapped("hi")); list(wrapped("again"))
    assert turns == ["hi", "again"]
    assert len(notes) == 1 and notes[0].startswith("the clock says ")   # same minute: one note
