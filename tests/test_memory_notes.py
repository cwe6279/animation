import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker.memory_notes import Notebook, TaskLedger


def test_note_lands_under_todays_heading_once(tmp_path):
    nb = Notebook(str(tmp_path))
    d = datetime(2026, 9, 12, 14, 2)
    nb.note("the board moved to Thursday", now=d)
    nb.note("  Sam prefers   email ", now=d.replace(minute=10))
    text = open(tmp_path / "notes.md", encoding="utf-8").read()
    assert text.count("## 2026-09-12") == 1
    assert "- 14:02 — the board moved to Thursday" in text
    assert "- 14:10 — Sam prefers email" in text
    # a new day gets its own heading
    nb.note("next day", now=datetime(2026, 9, 13, 9, 0))
    text = open(tmp_path / "notes.md", encoding="utf-8").read()
    assert text.index("## 2026-09-13") > text.index("## 2026-09-12")


def test_recent_drops_the_title_and_truncates_from_the_front(tmp_path):
    nb = Notebook(str(tmp_path))
    assert nb.recent() == "(no notes yet)"
    for i in range(40):
        nb.note(f"item number {i} with some padding text", now=datetime(2026, 9, 12, 8, i))
    full = nb.recent(max_chars=100000)
    assert not full.startswith("# ")
    assert "item number 39" in full
    short = nb.recent(max_chars=200)
    assert short.startswith("… (older notes omitted)")
    assert "item number 39" in short and "item number 0 " not in short
    assert len(short) < 260


def test_session_summary_is_appended(tmp_path):
    nb = Notebook(str(tmp_path))
    nb.session_summary("We planned the offsite.\nFollow up with Sam.", now=datetime(2026, 9, 12, 17, 0))
    text = open(tmp_path / "notes.md", encoding="utf-8").read()
    assert "### Session summary 17:00" in text and "Follow up with Sam." in text
    nb.session_summary("   ")            # empty: nothing written
    assert text == open(tmp_path / "notes.md", encoding="utf-8").read()


def test_ledger_states_survive_a_reload(tmp_path):
    led = TaskLedger(str(tmp_path))
    assert led.render() == "(no tasks yet)"
    led.add("3f2a", "research the  supplier delays", now=datetime(2026, 9, 12, 14, 2))
    led.add("1a2b", "find three projectors", now=datetime(2026, 9, 12, 14, 5))
    assert [it["id"] for it in led.open_items()] == ["3f2a", "1a2b"]
    assert led.set_state("3f2a", "in progress")
    assert led.set_state("1a2b", "done", summary="The Epson is the pick.\nTwo others listed.")
    assert not led.set_state("zzzz", "done")
    again = TaskLedger(str(tmp_path))
    assert again.get("3f2a")["state"] == "in progress"
    assert again.get("1a2b")["summary"] == "The Epson is the pick. Two others listed."
    text = again.render()
    assert "- [in progress] 3f2a · 2026-09-12 14:02 · research the supplier delays" in text
    assert "- [done] 1a2b" in text and "    The Epson is the pick." in text
    assert [it["id"] for it in again.open_items()] == ["3f2a"]


def test_ledger_render_keeps_open_tasks_and_the_newest_finished(tmp_path):
    led = TaskLedger(str(tmp_path))
    for i in range(30):
        led.add(f"d{i:02d}", f"finished job {i}", now=datetime(2026, 9, 1, 8, i))
        led.set_state(f"d{i:02d}", "done", summary="ok")
    led.add("open1", "still going", now=datetime(2026, 9, 12, 9, 0))
    text = led.render(max_chars=400)
    assert "still going" in text
    assert "finished job 29" in text
    assert "finished job 0\n" not in text
    assert text.startswith("… (")
    import pytest
    with pytest.raises(ValueError):
        led.set_state("open1", "finished")
