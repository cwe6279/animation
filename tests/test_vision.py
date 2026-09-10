import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from talker.vision import SceneWatcher, SceneNote, resolve_camera


class FakeSource:
    def __init__(self): self.calls = 0
    def burst(self, n): self.calls += 1; return [b"jpg%d" % i for i in range(n)]
    def close(self): pass


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def test_watcher_keeps_only_latest_notes_and_builds_context(tmp_path):
    clk = Clock()
    replies = iter([{"state": "two kids in costumes", "changes": "two kids arrived", "people": 2},
                    {"state": "empty room", "changes": "the kids left", "people": 0},
                    {"state": "an adult waving", "changes": "an adult came in, waving", "people": 1},
                    {"state": "an adult and a dog", "changes": "a dog appeared", "people": 1}])
    seen_previous = []
    def describe(frames, previous):
        seen_previous.append(previous); return next(replies)
    w = SceneWatcher(FakeSource(), describe, keep=2, emergency_dir=str(tmp_path), clock=clk)
    for _ in range(4):
        w.observe_once()
    assert seen_previous[1] == "two kids in costumes"           # the model gets the last state to diff against
    assert [n.notes for n in w._notes] == ["an adult waving", "an adult and a dog"]
    ctx = w.context(max_age_s=40)
    assert ctx == "a dog appeared"                                # only the delta reaches the brain
    clk.t += 100                       # stale notes are not offered
    assert w.context(max_age_s=40) == ""


def test_emergency_saves_frames_and_note(tmp_path):
    w = SceneWatcher(FakeSource(), lambda f: {"state": "child on floor", "changes": "child fell, crying", "people": 1,
                                             "emergency": True, "emergency_reason": "child on the floor crying"},
                     emergency_dir=str(tmp_path), on_error=lambda m: None)
    note = w.observe_once()
    assert note.emergency and w.stats["emergencies"] == 1
    folders = list(tmp_path.iterdir()); assert len(folders) == 1
    files = sorted(p.name for p in folders[0].iterdir())
    assert files == ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg", "note.json"]
    assert json.load(open(folders[0] / "note.json"))["emergency_reason"].startswith("child")
    assert "EMERGENCY" in w.context()


def test_non_emergency_keeps_no_files(tmp_path):
    w = SceneWatcher(FakeSource(), lambda f: {"state": "quiet room", "changes": "no change", "people": 0}, emergency_dir=str(tmp_path))
    w.observe_once(); w.observe_once()
    assert not list(tmp_path.iterdir())


def test_no_change_gives_brain_nothing_but_first_look_gives_state(tmp_path):
    notes = []
    w = SceneWatcher(FakeSource(), lambda f, prev: {"state": "one adult at a desk", "changes": "no change", "people": 1},
                     emergency_dir=str(tmp_path), on_note=notes.append, signature=None)
    w.observe_once()
    assert w.context() == "one adult at a desk" and len(notes) == 1   # first look: state
    w.observe_once()
    assert w.context() == "" and len(notes) == 1                             # nothing new: silence


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
    from talker.brains.claude_chat import ClaudeChat
    from tests.test_claude_chat import FakeClient
    client = FakeClient(["ok"])
    chat = ClaudeChat(client=client)
    chat.add_context("What you can see right now: two kids in costumes")
    list(chat.reply("hello"))
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "user" and msgs[0]["content"].startswith("(You notice: What you can see")
    assert msgs[1] == {"role": "user", "content": "hello"}       # visitor's words untouched
    list(chat.reply("and again"))
    msgs = client.calls[1]["messages"]
    assert sum("(You notice" in m["content"] for m in msgs if m["role"] == "user") == 1   # no repeat on a quiet turn


def test_unchanged_scene_skips_the_model_but_keeps_note_fresh(tmp_path):
    import numpy as np
    clk = Clock()
    sigs = iter([np.zeros((14, 24)), np.zeros((14, 24)) + 0.01, np.zeros((14, 24)) + 0.5, np.zeros((14, 24)) + 0.5])
    calls = []
    w = SceneWatcher(FakeSource(), lambda f: calls.append(1) or {"state": f"state {len(calls)}", "changes": f"change {len(calls)}", "people": 1},
                     emergency_dir=str(tmp_path), clock=clk, signature=lambda jpg: next(sigs))
    w.observe_once(); clk.t += 9
    w.observe_once(); clk.t += 9                 # ~same signature: skipped, note time refreshed
    assert len(calls) == 1 and w.stats["skipped_unchanged"] == 1
    assert w.latest().time == clk.t - 9 and "change 1" in w.context()
    w.observe_once(); clk.t += 9                 # big change: described again
    assert len(calls) == 2 and "change 2" in w.context()
    w.observe_once()                             # same as last described: skipped
    assert len(calls) == 2


def test_heartbeat_describes_after_max_quiet(tmp_path):
    import numpy as np
    clk = Clock(); calls = []
    w = SceneWatcher(FakeSource(), lambda f: calls.append(1) or {"state": "still", "changes": "no change", "people": 0},
                     emergency_dir=str(tmp_path), clock=clk, max_quiet_s=30, signature=lambda jpg: np.zeros((14, 24)))
    w.observe_once()
    for _ in range(5):
        clk.t += 9; w.observe_once()
    assert len(calls) == 2                       # one at start, one heartbeat after 30 s of no change


def test_request_wakes_watcher_and_wait_for_returns(tmp_path):
    import threading
    calls = []
    w = SceneWatcher(FakeSource(), lambda f: calls.append(1) or {"state": "x", "changes": "x", "people": 0},
                     interval=60, emergency_dir=str(tmp_path))
    w.start()
    assert w.wait_for(1, timeout=2)                  # first tick
    ticket = w.request(force=True)
    assert w.wait_for(ticket, timeout=2) and len(calls) == 2
    w.stop()


def test_loop_requests_burst_when_visitor_starts_talking():
    from talker.voice_loop import VoiceLoop
    from talker.stt_backends import STTBackend, Transcript

    class STT(STTBackend):
        speech_active = False
        def feed(self, pcm): return None
        def reset(self): pass

    class FakeVision:
        def __init__(self): self.requests = 0
        def request(self, force=False): self.requests += 1; return self.requests
        def wait_for(self, ticket, timeout): return True

    class Spk:
        is_busy = False
        def speak_stream(self, c): list(c)
        def interrupt(self): pass
    stt, vis = STT(), FakeVision()
    loop = VoiceLoop(stt, lambda t: iter(["ok"]), Spk(), on_event=lambda k, s: None)
    loop.vision = vis
    loop._process(b"\x00" * 320)                    # silence
    stt.speech_active = True
    loop._process(b"\x00" * 320); loop._process(b"\x00" * 320)   # talking (two chunks, one request)
    assert vis.requests == 1
    stt.speech_active = False
    loop._process(b"\x00" * 320)
    stt.speech_active = True
    loop._process(b"\x00" * 320)                    # a new utterance -> a new request
    assert vis.requests == 2


def test_trivial_deltas_are_dropped():
    from talker.vision import _trivial_change
    assert _trivial_change("Person's hand position changed slightly; no significant new objects or arrivals.")
    assert _trivial_change("no change") and _trivial_change("No change.") and _trivial_change("")
    assert not _trivial_change("A child came in holding a red balloon.")
    assert not _trivial_change("The visitor is now wearing a maroon cap.")


def test_readable_text_is_reported_once_when_it_appears():
    """New legible text is a notice on its own; the same text is not repeated next look."""
    replies = iter([{"state": "one adult", "changes": "an adult came in", "people": 1, "text": ""},
                    {"state": "one adult holding a sign", "changes": "no change", "people": 1, "text": "FREE CANDY"},
                    {"state": "one adult holding a sign", "changes": "no change", "people": 1, "text": "FREE CANDY"}])
    notes = []
    w = SceneWatcher(FakeSource(), lambda f, prev: next(replies), on_note=notes.append, signature=None)
    w.observe_once()
    w.observe_once()
    assert len(notes) == 2 and 'Readable text: "FREE CANDY"' in w.context()
    w.observe_once()
    assert len(notes) == 2 and w.context() == ""        # same text again: nothing new to say


def test_look_now_forces_a_description_a_still_room_would_skip():
    """A visual question must get a picture even when nothing moved."""
    seen = []

    def describe(frames, prev):
        seen.append(prev)
        return {"state": "one adult at a desk", "changes": "no change", "people": 1}

    import numpy as np
    still = np.zeros((4, 4), dtype=np.float32)                 # every burst looks identical
    w = SceneWatcher(FakeSource(), describe, on_note=lambda n: None, signature=lambda jpg: still)
    w.observe_once(force=True)                       # the first look, as at startup
    assert len(seen) == 1 and w.context() == "one adult at a desk"
    w.observe_once()                                 # a still room: skipped by the change gate
    assert len(seen) == 1 and w.stats["skipped_unchanged"] == 1
    w.start()
    try:
        assert w.look_now(timeout=5) == "one adult at a desk"   # forced, and it returns the state
    finally:
        w.stop()
    assert len(seen) == 2
