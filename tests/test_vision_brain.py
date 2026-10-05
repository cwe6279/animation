"""Vision spends little while dormant and backs off after failures; the brain says why it failed."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker.vision import SceneWatcher, make_describer


class Src:
    def __init__(self): self.bursts = 0
    def burst(self, n=3, spacing_s=0.0):
        self.bursts += 1
        return [b"jpeg-%d" % self.bursts]
    def close(self): pass


def watcher(describe, **kw):
    return SceneWatcher(Src(), describe, interval=9.0, signature=None, **kw)


def test_the_periodic_pace_is_slow_while_dormant_and_backs_off_after_failures():
    w = watcher(lambda f, p="": {"state": "a desk"}, dormant_interval=300)
    assert w.next_wait() == 9.0
    w.is_dormant = lambda: True
    assert w.next_wait() == 300
    w._fail_streak = 1
    assert w.next_wait() == 30
    w._fail_streak = 4
    assert w.next_wait() == 300
    w._fail_streak = 99
    assert w.next_wait() == 600


def test_failures_are_logged_sparingly_and_a_success_resets_the_backoff():
    calls, errors = {"n": 0, "fail": True}, []

    def describe(frames, prev=""):
        calls["n"] += 1
        if calls["fail"]:
            raise RuntimeError("Error code: 400 - credit balance is too low")
        return {"state": "a desk", "changes": "a person arrived", "people": 1}
    w = watcher(describe, on_error=errors.append)
    w.backoff_steps = (0.01, 0.01, 0.01, 0.01, 0.01)
    w.start()
    deadline = time.time() + 3
    while calls["n"] < 6 and time.time() < deadline:
        w.request(force=True)
        time.sleep(0.03)
    assert w._fail_streak >= 5 and len(errors) <= 3      # 1st, 3rd, 5th failures are logged, not every one
    assert "next look in" in errors[0]
    calls["fail"] = False
    while w._fail_streak and time.time() < deadline + 2:
        w.request(force=True)
        time.sleep(0.03)
    w.stop()
    assert w._fail_streak == 0 and w.latest().notes == "a desk"


def test_sound_cannot_make_a_dormant_watcher_look_more_than_once_a_minute():
    looks = []
    w = watcher(lambda f, p="": looks.append(1) or {"state": "a desk"}, dormant_min_gap=60,
                is_dormant=lambda: True, dormant_interval=300)
    w.start()
    deadline = time.time() + 1.5
    while time.time() < deadline:
        w.request(force=False)                         # what a sound onset does
        time.sleep(0.05)
    assert len(looks) == 1 and w.stats["dormant_skips"] > 3
    t = threading.Thread(target=lambda: w.look_now(timeout=1.0))
    t.start(); t.join()
    w.stop()
    assert len(looks) == 2                               # a visual question is always served


def test_auto_backend_uses_the_local_model_and_falls_back_to_claude(monkeypatch):
    import talker.vision as v
    seen = []
    monkeypatch.setattr(v, "describe_with_ollama", lambda f, p="", **k: seen.append("local") or {"state": "local"})
    monkeypatch.setattr(v, "describe_with_claude", lambda f, p="", **k: seen.append("claude") or {"state": "claude"})
    monkeypatch.setenv("ANTHROPIC_VISION_API_KEY", "sk-test")
    d = make_describer("auto")
    assert d([b"x"])["state"] == "local"

    def down(f, p="", **k):
        raise OSError("connection refused")
    monkeypatch.setattr(v, "describe_with_ollama", down)
    assert d([b"x"])["state"] == "claude" and seen == ["local", "claude"]
    assert make_describer("anthropic")([b"x"])["state"] == "claude"


def test_brain_failures_say_why():
    from talker.voice_loop import brain_problem
    assert brain_problem(RuntimeError("Error code: 400 - Your credit balance is too low"))[0] == "out_of_credit"
    assert "out of credit" in brain_problem(RuntimeError("credit balance is too low"))[1]
    assert brain_problem(RuntimeError("Error code: 401 - invalid x-api-key"))[0] == "bad_key"
    assert brain_problem(RuntimeError("Error code: 429 rate_limit_error"))[0] == "rate_limited"
    assert brain_problem(RuntimeError("Error code: 529 overloaded_error"))[0] == "overloaded"
    assert brain_problem(OSError("Connection error."))[0] == "unreachable"
    assert brain_problem(ValueError("something odd"))[0] == "error"


def test_brain_health_records_each_change_once():
    from talker.voice_loop import VoiceLoop
    seen = []
    loop = VoiceLoop.__new__(VoiceLoop)
    loop.on_brain = lambda ok, kind, detail: seen.append((ok, kind))
    loop.brain_health(False, "out_of_credit", "400")
    loop.brain_health(False, "out_of_credit", "400")
    assert seen == [(False, "out_of_credit")] and loop.brain["ok"] is False
    loop.brain_health(True)
    loop.brain_health(True)
    assert seen == [(False, "out_of_credit"), (True, "")] and loop.brain["ok"] is True


def test_on_demand_looks_only_when_asked():
    looks = []
    w = watcher(lambda f, p="": looks.append(1) or {"state": "a desk"}, on_demand=True)
    assert w.next_wait() > 3600
    w.start()
    time.sleep(0.3)
    assert looks == []                                  # no look of its own
    w.request(force=False)                              # a sound onset
    time.sleep(0.3)
    assert looks == []
    w.request(force=True)                               # startup / waking up
    deadline = time.time() + 2
    while not looks and time.time() < deadline:
        time.sleep(0.02)
    assert looks == [1]
    assert w.look_now(timeout=2.0) == "a desk" and len(looks) == 2
    w.stop()
