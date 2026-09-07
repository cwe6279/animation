import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from vision import SceneWatcher, SceneNote, resolve_camera


class FakeSource:
    def __init__(self): self.calls = 0
    def burst(self, n): self.calls += 1; return [b"jpg%d" % i for i in range(n)]
    def close(self): pass


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def test_watcher_keeps_only_latest_notes_and_builds_context(tmp_path):
    clk = Clock()
    replies = iter([{"notes": "two kids in costumes", "people": 2}, {"notes": "nobody in view", "people": 0},
                    {"notes": "an adult waving", "people": 1}, {"notes": "a dog", "people": 0}])
    w = SceneWatcher(FakeSource(), lambda f: next(replies), keep=2, emergency_dir=str(tmp_path), clock=clk)
    for _ in range(4):
        w.observe_once()
    assert [n.notes for n in w._notes] == ["an adult waving", "a dog"]
    ctx = w.context(max_age_s=40)
    assert "a dog" in ctx and "camera notes" in ctx
    clk.t += 100                       # stale notes are not offered
    assert w.context(max_age_s=40) == ""


def test_emergency_saves_frames_and_note(tmp_path):
    w = SceneWatcher(FakeSource(), lambda f: {"notes": "child fell, crying", "people": 1, "emergency": True,
                                             "emergency_reason": "child on the floor crying"},
                     emergency_dir=str(tmp_path), on_error=lambda m: None)
    note = w.observe_once()
    assert note.emergency and w.stats["emergencies"] == 1
    folders = list(tmp_path.iterdir()); assert len(folders) == 1
    files = sorted(p.name for p in folders[0].iterdir())
    assert files == ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg", "note.json"]
    assert json.load(open(folders[0] / "note.json"))["emergency_reason"].startswith("child")
    assert "EMERGENCY" in w.context()


def test_non_emergency_keeps_no_files(tmp_path):
    w = SceneWatcher(FakeSource(), lambda f: {"notes": "quiet room", "people": 0}, emergency_dir=str(tmp_path))
    w.observe_once(); w.observe_once()
    assert not list(tmp_path.iterdir())


def test_describe_errors_are_counted_not_fatal(tmp_path):
    errors = []
    def boom(f): raise RuntimeError("api down")
    w = SceneWatcher(FakeSource(), boom, interval=0.01, emergency_dir=str(tmp_path), on_error=errors.append)
    w.start(); time.sleep(0.15); w.stop()
    assert w.stats["errors"] >= 1 and errors and w.context() == ""


def test_resolve_camera_accepts_index():
    assert resolve_camera(0) == 0 and resolve_camera("2") == 2


def test_chat_inserts_scene_context_only_when_pushed():
    pytest.importorskip("anthropic")
    from llm_integration.claude_chat import ClaudeChat
    from tests.test_claude_chat import FakeClient
    client = FakeClient(["ok"])
    chat = ClaudeChat(client=client)
    chat.add_context("What you can see right now: two kids in costumes")
    list(chat.reply("hello"))
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "user" and msgs[0]["content"].startswith("[Context, not spoken by anyone: What you can see")
    assert msgs[1] == {"role": "user", "content": "hello"}       # visitor's words untouched
    list(chat.reply("and again"))
    msgs = client.calls[1]["messages"]
    assert sum("[Context" in m["content"] for m in msgs if m["role"] == "user") == 1   # no repeat on a quiet turn


def test_unchanged_scene_skips_the_model_but_keeps_note_fresh(tmp_path):
    import numpy as np
    clk = Clock()
    sigs = iter([np.zeros((14, 24)), np.zeros((14, 24)) + 0.01, np.zeros((14, 24)) + 0.5, np.zeros((14, 24)) + 0.5])
    calls = []
    w = SceneWatcher(FakeSource(), lambda f: calls.append(1) or {"notes": f"note {len(calls)}", "people": 1},
                     emergency_dir=str(tmp_path), clock=clk, signature=lambda jpg: next(sigs))
    w.observe_once(); clk.t += 9
    w.observe_once(); clk.t += 9                 # ~same signature: skipped, note time refreshed
    assert len(calls) == 1 and w.stats["skipped_unchanged"] == 1
    assert w.latest().time == clk.t - 9 and "note 1" in w.context()
    w.observe_once(); clk.t += 9                 # big change: described again
    assert len(calls) == 2 and "note 2" in w.context()
    w.observe_once()                             # same as last described: skipped
    assert len(calls) == 2


def test_heartbeat_describes_after_max_quiet(tmp_path):
    import numpy as np
    clk = Clock(); calls = []
    w = SceneWatcher(FakeSource(), lambda f: calls.append(1) or {"notes": "still", "people": 0},
                     emergency_dir=str(tmp_path), clock=clk, max_quiet_s=30, signature=lambda jpg: np.zeros((14, 24)))
    w.observe_once()
    for _ in range(5):
        clk.t += 9; w.observe_once()
    assert len(calls) == 2                       # one at start, one heartbeat after 30 s of no change
