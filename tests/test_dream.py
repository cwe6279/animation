"""Dream mode: the digest is small and useful, the report is written, it runs once a day when idle."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker import dream as d


def test_condense_keeps_useful_lines_and_collapses_repeats():
    lines = ["19:00:01 [hearing] hel\n", "19:00:02 [you] hello\n", "19:00:03 [bot] hi there\n",
             "19:00:04 [stt] scribe connection lost (DNS); reconnecting in 15s\n",
             "19:00:19 [stt] scribe connection lost (DNS); reconnecting in 15s\n",
             "19:00:34 [stt] scribe connection lost (DNS); reconnecting in 15s\n",
             "W 19:00:35 mod.rt RTKit error\n", "pygame-ce 2.5.8\n", "19:01:00 [you] bye\n"]
    out = d.condense(lines)
    assert out[0] == "19:00:02 [you] hello" and out[1] == "19:00:03 [bot] hi there"
    assert out[2].endswith("(repeated x3)") and out[3] == "19:01:00 [you] bye" and len(out) == 4


def test_turn_stats_and_trim():
    lines = ["[turn] you stopped -> transcript 900 ms -> first token 1900 ms -> first sentence to TTS 2000 ms -> first audio 2200 ms",
             "[turn] you stopped -> transcript 3400 ms -> first token ? ms -> first audio 5600 ms"]
    st = d.turn_stats(lines)
    assert st["turns"] == 2 and st["avg_first_audio_ms"] == 3900 and st["slow_turns_over_5s"] == 1
    assert st["avg_first_token_ms"] == 1900
    long = "x" * 1000
    t = d.trim(long, 100)
    assert "left out" in t and len(t) < 300


def _logs(tmp_path):
    logs, dreams, face = tmp_path / "logs", tmp_path / "dreams", tmp_path / "face"
    logs.mkdir(); face.mkdir()
    (logs / "voice-20261007-195750.log").write_text(
        "19:58:40 [voice] brain: claude claude-haiku-5-5\n20:01:10 [you] what is that?\n"
        "20:01:11 [bot] {{look}} Let me see.\n"
        "20:01:11 [turn] you stopped -> transcript 3400 ms -> first token 4400 ms -> first audio 4800 ms\n")
    (logs / "incidents.jsonl").write_text('{"time": "2026-10-07T20:01:11", "kind": "slow_turn", "first_audio_ms": 4800}\n')
    (face / "tasks.md").write_text("# Tasks\n- [done] g1 · flights\n")
    (face / "face.json").write_text('{"name": "clara", "models": {"claude": "claude-sonnet-5"}, "voices": {"x": "secret"}}')
    return logs, dreams, face


def test_build_digest_gathers_logs_incidents_and_context(tmp_path):
    logs, dreams, face = _logs(tmp_path)
    dg = d.build_digest(0, str(face), logs_dir=str(logs), dreams_dir=str(dreams))
    assert dg["sessions"] == 1 and dg["incidents"] == 1
    assert "[you] what is that?" in dg["text"] and "avg_first_audio_ms" in dg["text"]
    assert "tasks.md" in dg["text"] and '"models"' in dg["text"] and "secret" not in dg["text"]


class FakeStream:
    def __init__(self, text): self.text = text
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self):
        class B: type, text = "text", None
        b = B(); b.text = self.text
        class U: input_tokens, output_tokens = 12000, 900
        class M: content, stop_reason, usage = [b], "end_turn", U()
        return M()


class FakeClient:
    def __init__(self): self.calls = []
    def with_options(self, **k): return self
    @property
    def beta(self): return self
    @property
    def messages(self): return self
    def stream(self, **k):
        self.calls.append(k)
        return FakeStream("## Summary\nMostly fine; the look results were dropped.")


def test_dream_writes_a_report_and_remembers_what_it_covered(tmp_path):
    logs, dreams, face = _logs(tmp_path)
    client = FakeClient()
    now = time.mktime(time.strptime("2026-10-08 03:00", "%Y-%m-%d %H:%M"))
    path = d.dream(face_dir=str(face), client=client, now=now, logs_dir=str(logs), dreams_dir=str(dreams))
    assert path.endswith("2026-10-08.md")
    text = open(path, encoding="utf-8").read()
    assert "look results were dropped" in text and "12000 tokens in" in text
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default" and "thinking" not in call
    state = d.load_state(str(dreams / "state.json"))
    assert state["last_dream"] == "2026-10-08" and state["covered_until"] == now


def test_scheduler_dreams_once_a_day_only_when_idle_and_dormant(tmp_path):
    clock = {"t": time.mktime(time.strptime("2026-10-08 03:00", "%Y-%m-%d %H:%M"))}
    st = {"dormant": True, "last": clock["t"] - 4 * 3600, "online": True}
    sch = d.DreamScheduler(run=lambda: None, is_dormant=lambda: st["dormant"], last_activity=lambda: st["last"],
                           idle_hours=3, online=lambda: st["online"], clock=lambda: clock["t"],
                           state_path=str(tmp_path / "state.json"))
    assert sch.due()
    st["dormant"] = False
    assert not sch.due()
    st["dormant"], st["last"] = True, clock["t"] - 3600
    assert not sch.due()                                  # talked an hour ago
    st["last"] = clock["t"] - 5 * 3600
    st["online"] = False
    assert not sch.due()
    st["online"] = True
    d.save_state({"last_dream": "2026-10-08"}, str(tmp_path / "state.json"))
    assert not sch.due()                                  # already dreamed today
