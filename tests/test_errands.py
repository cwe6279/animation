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


def test_capabilities_come_from_the_backend_when_it_has_them():
    calls = []
    def fetch(url, method="GET", body=None, timeout=5.0):
        calls.append(url)
        if url.endswith("/capabilities"):
            return {"can": ["search the web", " read the calendar ", ""]}
        raise RuntimeError("nope")
    r = ErrandRunner("http://box:8030", fetch=fetch)
    assert r.capabilities() == "search the web, read the calendar"
    assert ErrandRunner("http://box:8030", fetch=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))).capabilities() == ""
    assert ErrandRunner("http://box:8030", fetch=lambda *a, **k: {"can": "run code"}).capabilities() == "run code"
