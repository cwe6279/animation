"""Resilience: the network watchdog, the incident record, the always-speak fallback, stamped logs."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker import incidents
from talker.netwatch import NetWatch


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def test_offline_after_two_misses_then_online_with_the_outage_length(capsys):
    up = {"ok": True}
    events, clock = [], Clock()
    net = NetWatch(probe=lambda: up["ok"], on_offline=lambda: events.append("off"),
                   on_online=lambda s: events.append(("on", s)), heal_cmd=None, clock=clock)
    net.check()
    up["ok"] = False
    net.check()
    assert net.online and events == []                  # one miss is a blip
    net.check()
    assert not net.online and events == ["off"] and net.outages == 1
    clock.t += 95
    net.check()
    assert events == ["off"]                            # said once, not every check
    up["ok"] = True
    net.check()
    assert net.online and events[-1] == ("on", 95)


def test_heal_command_runs_while_offline_at_most_once_per_interval(monkeypatch):
    runs, clock = [], Clock()
    import talker.netwatch as nw

    class R:
        stdout, stderr, returncode = "Device 'wlan0' successfully activated.", "", 0
    monkeypatch.setattr(nw.subprocess, "run", lambda cmd, **k: runs.append(cmd) or R())
    net = NetWatch(probe=lambda: False, heal_after=120, clock=clock)   # the default heal command
    net.check(); net.check()                            # offline now
    clock.t += 60; net.check()
    assert runs == []
    clock.t += 61; net.check()
    # nmcli has no "device reconnect" (every heal failed); "device connect" re-activates wlan0
    assert runs == [["nmcli", "--wait", "25", "device", "connect", "wlan0"]]
    clock.t += 10; net.check()
    assert len(runs) == 1


def test_incidents_are_json_lines_with_time_and_kind(tmp_path):
    p = str(tmp_path / "incidents.jsonl")
    incidents.record("net_offline", path=p)
    incidents.record("slow_turn", path=p, first_audio_ms=7612)
    got = incidents.recent(path=p)
    assert [g["kind"] for g in got] == ["net_offline", "slow_turn"]
    assert got[1]["first_audio_ms"] == 7612 and "T" in got[1]["time"]


def test_a_reply_of_only_blocks_gets_a_spoken_word():
    from talker.voice_loop import always_speaks
    notes_only = always_speaks(lambda t: iter(["{{note Bluetooth is choppy; test a wired speaker.}}"]))
    assert "".join(notes_only("x")).endswith("Noted.")
    spoken = always_speaks(lambda t: iter(["[calm] Got it. ", "{{note birthday April 2}}"]))
    assert "Noted" not in "".join(spoken("x"))
    tag_only = always_speaks(lambda t: iter(["[calm] "]))
    assert "".join(tag_only("x")).endswith("Noted.")
    empty = always_speaks(lambda t: iter([]))
    assert "".join(empty("x")) == ""


def test_session_log_lines_are_stamped_in_the_file_only():
    from talker.session_log import _Tee
    term, fh = io.StringIO(), io.StringIO()
    tee = _Tee(term, fh)
    tee.write("[you] hello\n[bot] hi")
    tee.write(" there\n")
    assert term.getvalue() == "[you] hello\n[bot] hi there\n"
    lines = fh.getvalue().splitlines()
    assert len(lines) == 2 and lines[0][8:] == " [you] hello" and lines[1].endswith("[bot] hi there")
    assert lines[0][2] == ":" and lines[1][5] == ":"


def _run(chunks):
    from talker.voice_loop import handoff_brief
    return "".join(handoff_brief(lambda t: iter(chunks))("x"))


def test_a_handoff_gets_an_instant_short_ack_and_no_readback():
    from talker.voice_loop import handoff_brief
    chunks = ["{{ta", "sk search Delta flights after 4pm from NYC to Savannah, then text the list to "
              "+1 212 555 0199; the owner asked}", "} I'll search Delta flights after 4pm and text them to you."]
    stream = handoff_brief(lambda t: iter(chunks))("x")
    first = next(stream)
    assert first == "On it. "                               # spoken before the block finished streaming
    rest = "".join(stream)
    assert rest.startswith("{{task search Delta") and rest.endswith("the owner asked}}")
    assert "I'll search" not in rest                        # the readback is dropped


def test_an_emotion_tag_before_the_block_is_kept_and_later_blocks_still_pass():
    out = _run(["[calm] {{approve g1a2}} Sending it now. {{note owner approved the text}} Done."])
    assert out.startswith("[calm] Okay, going ahead. {{approve g1a2}}")
    assert "{{note owner approved the text}}" in out and "Sending it now" not in out and "Done." not in out


def test_ordinary_replies_stream_through_unchanged():
    chunks = ["Sure, ", "Hilton Head is 74 to 80 today. ", "{{task look it up}}"]
    assert _run(chunks) == "".join(chunks)                  # block not first: nothing is cut
    assert _run(["{{note birthday April 2}} Got it."]) == "{{note birthday April 2}} Got it."
    assert _run(["{"]) == "{"                               # too short to decide: kept
    assert _run(["{{deny g9}}"]) == "Alright, I won't. {{deny g9}}"
