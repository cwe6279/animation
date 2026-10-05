import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker.errands import ErrandRunner


class FakeBackend:
    """Stands in for tools/agent_relay.py: a dict of tasks and a switch to be 'down'."""
    def __init__(self):
        self.tasks = {}
        self.down = False
        self.calls = []

    def fetch(self, url, method="GET", body=None, timeout=5.0):
        if url.endswith("/health"):                 # protocol discovery: a version 1 agent
            if self.down:
                raise RuntimeError(f"{url} not reachable: refused")
            return {"ok": True}
        self.calls.append((method, url))
        if self.down:
            raise RuntimeError(f"{url} not reachable: refused")
        if method == "POST":
            tid = "srv-" + body["id"]
            self.tasks[tid] = {"state": "running", "summary": "", "result": "", "task": body["task"]}
            return {"id": tid}
        tid = url.rsplit("/", 1)[1]
        return self.tasks[tid]

    def finish(self, tid, summary="Done: the Epson is the pick.", result="Long form ..."):
        self.tasks[tid].update(state="done", summary=summary, result=result)


def test_submit_returns_at_once_and_a_done_task_fires_once():
    be = FakeBackend()
    done = []
    r = ErrandRunner("http://box:8030/", on_done=done.append, fetch=be.fetch)
    t0 = time.perf_counter()
    e = r.submit("find  three projectors", context="budget 300")
    assert time.perf_counter() - t0 < 0.01
    assert e.state == "queued" and e.task == "find three projectors"
    assert be.calls == []                       # nothing happened on the caller's thread
    r.cycle()
    assert e.state == "running" and e.remote_id == "srv-" + e.id
    assert be.calls[0] == ("POST", "http://box:8030/tasks")
    r.cycle()
    assert done == []
    be.finish(e.remote_id)
    r.cycle(); r.cycle()
    assert done == [e] and e.state == "done" and e.summary.startswith("Done:")
    assert r.stats["done"] == 1 and r.pending() == []


def test_a_backend_that_is_down_loses_nothing():
    be = FakeBackend()
    be.down = True
    r = ErrandRunner("http://box:8030", fetch=be.fetch)
    a = r.submit("task a"); b = r.submit("task b")
    r.cycle(); r.cycle()
    assert a.state == b.state == "queued" and r.reachable is False
    be.down = False
    r.cycle()
    assert a.state == b.state == "running" and r.reachable is True
    be.down = True                              # down again while polling: still running, retried
    r.cycle()
    assert a.state == "running"
    be.down = False
    be.finish(a.remote_id); be.finish(b.remote_id, summary="")
    r.cycle()
    assert a.state == b.state == "done"


def test_failed_task_and_announcements_wait_for_a_quiet_room():
    be = FakeBackend()
    failed, said = [], []
    quiet = {"ok": False}
    r = ErrandRunner("http://box:8030", on_fail=failed.append,
                     deliver=lambda t: said.append(t) or True if quiet["ok"] else False, fetch=be.fetch)
    e = r.submit("impossible thing")
    r.cycle()
    be.tasks[e.remote_id].update(state="failed", summary="The task failed: no network")
    r.cycle()
    assert failed == [e] and e.state == "failed" and r.stats["failed"] == 1
    r.say_later("first"); r.say_later("second")
    r.cycle(); r.cycle()
    assert said == []                           # the room is busy: kept, in order
    quiet["ok"] = True
    r.cycle()
    assert said == ["first", "second"]
    r.cycle()
    assert said == ["first", "second"]          # delivered once


def test_thread_runs_and_stops():
    be = FakeBackend()
    r = ErrandRunner("http://box:8030", poll_s=0.05, fetch=be.fetch)
    r.start()
    e = r.submit("x")
    t0 = time.time()
    while e.state != "running" and time.time() - t0 < 2:
        time.sleep(0.01)
    assert e.state == "running"
    r.stop()
    assert not r._thread.is_alive()


def test_no_address_yet_keeps_tasks_until_one_is_set():
    be = FakeBackend()
    r = ErrandRunner("", fetch=be.fetch)
    e = r.submit("find a projector")
    r.cycle(); r.cycle()
    assert e.state == "queued" and be.calls == []          # nowhere to send it: kept, nothing tried
    r.set_url("http://box:8030/")
    assert r.url == "http://box:8030" and r.reachable is None
    r.cycle()
    assert e.state == "running" and be.calls[0] == ("POST", "http://box:8030/tasks")


def test_health_reports_without_raising():
    be = FakeBackend()
    r = ErrandRunner("", fetch=lambda url, *a, **k: {"ok": True, "running": 2} if url.endswith("/health") else be.fetch(url, *a, **k))
    assert r.health()["ok"] is False and "no address" in r.health()["detail"]
    h = r.health("http://box:8030")
    assert h["ok"] is True and h["running"] == 2 and "ms" in h
    be.down = True
    r2 = ErrandRunner("http://box:8030", fetch=be.fetch)
    h = r2.health()
    assert h["ok"] is False and "not reachable" in h["detail"]


def test_agent_control_validates_saves_and_lists():
    from talker.errands import AgentControl
    be = FakeBackend()
    saved = []
    r = ErrandRunner("", fetch=be.fetch)
    ctl = AgentControl(r, can="research, write", source="none", save=saved.append)
    assert "error" in ctl.set_url("agentbox:8030")          # no scheme
    out = ctl.set_url("http://agentbox:8030/")
    assert out["url"] == "http://agentbox:8030" and saved == ["http://agentbox:8030"]
    r.submit("one"); r.submit("two")
    info = ctl.info()
    assert info["url"] == "http://agentbox:8030" and info["can"] == "research, write" and info["open"] == 2
    assert [t["task"] for t in info["tasks"]] in (["two", "one"], ["one", "two"])   # same-tick creation order
    ctl.set_url("", save=True)
    assert r.url == "" and saved[-1] == ""
    flagged = AgentControl(ErrandRunner("http://a:1", fetch=be.fetch), source="flag", save=saved.append)
    assert "--agent-url" in flagged.set_url("http://b:2")["message"]


def test_local_settings_round_trip(tmp_path):
    from talker import local_settings
    p = str(tmp_path / "settings.json")
    assert local_settings.load(p) == {}
    local_settings.save("agent_url", "http://agentbox:8030", p)
    assert local_settings.load(p) == {"agent_url": "http://agentbox:8030"}
    local_settings.save("agent_url", None, p)
    assert local_settings.load(p) == {}


def test_resume_polls_a_task_left_open_last_time():
    be = FakeBackend()
    be.tasks["e535"] = {"state": "running", "summary": "", "result": ""}
    done = []
    r = ErrandRunner("http://box:8030", on_done=done.append, fetch=be.fetch)
    e = r.resume("e535", "research mild winters")
    assert e.state == "running" and r.pending() == [e]
    r.cycle()
    assert done == []
    be.finish("e535", summary="Lisbon, Valletta and Athens.")
    r.cycle()
    assert done == [e] and e.summary.startswith("Lisbon")



def test_she_says_the_agent_is_unreachable_once_and_says_when_it_returns():
    """On a trip a handed-off task that silently queues looks ignored."""
    be = FakeBackend()
    said = []
    r = ErrandRunner("http://box:8030", deliver=lambda t: said.append(t) or True, fetch=be.fetch)
    be.down = True
    r.submit("find three projectors")
    r.cycle()
    assert len(said) == 1 and "cannot be reached" in said[0] and "task you handed over" in said[0]
    r.cycle(); r.cycle()
    assert len(said) == 1                       # once per change, not once per cycle
    be.down = False
    r.cycle()
    assert len(said) == 2 and "can be reached again" in said[1]
    assert r.pending() and r.pending()[0].state == "running"


def test_nothing_is_said_when_the_outbox_is_empty():
    be = FakeBackend()
    said = []
    r = ErrandRunner("http://box:8030", deliver=lambda t: said.append(t) or True, fetch=be.fetch)
    be.down = True
    r.cycle(); r.cycle()
    assert said == [] and r.reachable is not True
    be.down = False
    r.cycle()
    assert said == []                           # nothing was waiting, so there is nothing to report


def test_the_timeout_reaches_every_request():
    seen = []
    def fetch(url, method="GET", body=None, timeout=5.0):
        seen.append(timeout)
        return {"id": "srv-1"} if method == "POST" else {"ok": True, "state": "running"}
    r = ErrandRunner("http://box:8030", fetch=fetch, timeout=30.0)
    assert r.timeout == 30.0
    r.health(); r.submit("x"); r.cycle()
    assert seen and set(seen) == {30.0}
    assert ErrandRunner("http://box:8030").timeout == 10.0     # the default, raised for slow links
