"""
errands.py — hand long work to an agent on another machine and hear back later.

The character stays in the room: submitting is a queue append that returns at
once, and one background thread does the HTTP. The backend is whatever answers
this small contract (tools/agent_relay.py is one, wrapping any command-line
agent harness):

    POST {url}/tasks        {"id": "3f2a", "task": "...", "context": "...", "from": "clara"}
                            -> {"id": "3f2a"}          (the backend may assign its own id)
    GET  {url}/tasks/{id}   -> {"state": "queued|running|done|failed",
                                "summary": "...",      (two or three spoken sentences)
                                "result": "..."}       (the long form; written to the ledger)
    GET  {url}/capabilities -> {"can": ["search the web", ...]}   optional: what the agent can do,
                                told to the character so she hands those things off

Polling, not callbacks: it works through any firewall and needs no open port on
the character's machine. A backend that is down is retried every cycle and said
once in the log, never raised.

    runner = ErrandRunner("http://agentbox:8030", on_done=lambda e: ...)
    runner.start()
    runner.submit("find three 4K projectors under 300")     # from any thread, ~1 µs
    runner.say_later("Task 3f2a finished: ...")           # tried each cycle until deliver() takes it
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Errand:
    id: str                       # ours; the ledger and the log use it
    task: str
    context: str = ""
    remote_id: Optional[str] = None
    state: str = "queued"         # queued (not yet posted) | running | done | failed
    summary: str = ""
    result: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)


def _http(url: str, method: str = "GET", body: Optional[dict] = None, timeout: float = 5.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code} from {url}: {e.read().decode(errors='replace')[:200]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} not reachable: {e.reason}")
    return json.loads(raw) if raw.strip() else {}


class ErrandRunner:
    """One daemon thread: posts what was submitted, polls what is running, and
    delivers announcements when the room is quiet."""

    def __init__(self, url: str, poll_s: float = 5.0,
                 on_done: Optional[Callable[[Errand], None]] = None,
                 on_fail: Optional[Callable[[Errand], None]] = None,
                 on_started: Optional[Callable[[Errand], None]] = None,
                 deliver: Optional[Callable[[str], bool]] = None,
                 fetch: Callable[..., dict] = _http, sender: str = "talker"):
        self.url = url.rstrip("/")
        self.poll_s = poll_s
        self.on_done = on_done
        self.on_fail = on_fail
        self.on_started = on_started
        self.deliver = deliver          # deliver(text) -> True when it was said; else retried
        self.fetch = fetch
        self.sender = sender
        self.errands: Dict[str, Errand] = {}
        self._outbox: List[Errand] = []
        self._events: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reachable: Optional[bool] = None
        self.stats = {"submitted": 0, "done": 0, "failed": 0, "cycles": 0, "errors": 0}
        self.last_summary = ""

    def capabilities(self, timeout: float = 3.0) -> str:
        """What the backend says it can do, as one line; '' if it does not say."""
        try:
            resp = self.fetch(f"{self.url}/capabilities", timeout=timeout)
        except Exception:
            return ""
        can = resp.get("can") if isinstance(resp, dict) else resp
        if isinstance(can, (list, tuple)):
            can = ", ".join(str(x).strip() for x in can if str(x).strip())
        return str(can or "").strip()

    # ── any thread ─────────────────────────────────────
    def submit(self, task: str, context: str = "") -> Errand:
        """Queue a task; returns at once. The poller posts it on its next pass."""
        e = Errand(id=secrets.token_hex(2), task=" ".join(task.split()), context=context.strip())
        with self._lock:
            self.errands[e.id] = e
            self._outbox.append(e)
            self.stats["submitted"] += 1
        self._wake.set()
        return e

    def resume(self, task_id: str, task: str) -> Errand:
        """A task that was still open when the character last shut down: poll it again.
        Ours and the backend's ids are the same when the backend kept ours (agent_relay does)."""
        e = Errand(id=task_id, task=task, remote_id=task_id, state="running")
        with self._lock:
            self.errands[task_id] = e
        self._wake.set()
        return e

    def say_later(self, text: str) -> None:
        """Queue an announcement; tried each cycle until deliver() accepts it."""
        with self._lock:
            self._events.append(text)
        self._wake.set()

    def pending(self) -> List[Errand]:
        with self._lock:
            return [e for e in self.errands.values() if e.state in ("queued", "running")]

    # ── lifecycle ──────────────────────────────────────
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="errand-poller", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                self.cycle()
            except Exception as e:                   # never let the poller die
                self.stats["errors"] += 1
                print(f"[errands] cycle failed: {e}")
            self._wake.wait(max(0.5, self.poll_s))

    # ── one pass (public so tests can drive it without the thread) ──
    def cycle(self) -> None:
        self.stats["cycles"] += 1
        self._post_outbox()
        self._poll_running()
        self._deliver_events()

    def _say_reachability(self, ok: bool, err: str = "") -> None:
        if ok != self.reachable:
            print(f"[errands] backend at {self.url} {'reachable' if ok else 'unreachable, will retry'}"
                  + (f": {err}" if err else ""))
        self.reachable = ok

    def _post_outbox(self) -> None:
        with self._lock:
            todo = list(self._outbox)
        for e in todo:
            try:
                resp = self.fetch(f"{self.url}/tasks", "POST",
                                  {"id": e.id, "task": e.task, "context": e.context, "from": self.sender})
            except Exception as ex:
                self._say_reachability(False, str(ex))
                return                                # keep the rest queued; try next cycle
            self._say_reachability(True)
            e.remote_id = str(resp.get("id") or e.id)
            e.state, e.updated = "running", time.time()
            with self._lock:
                self._outbox.remove(e)
            print(f"[errands] {e.id} started: {e.task}")
            if self.on_started:
                self.on_started(e)

    def _poll_running(self) -> None:
        for e in self.pending():
            if e.state != "running":
                continue
            try:
                resp = self.fetch(f"{self.url}/tasks/{e.remote_id}")
            except Exception as ex:
                self._say_reachability(False, str(ex))
                return
            self._say_reachability(True)
            state = str(resp.get("state", "running")).lower()
            if state in ("done", "failed"):
                e.state, e.updated = state, time.time()
                e.summary = str(resp.get("summary") or "").strip()
                e.result = str(resp.get("result") or "").strip()
                if state == "done":
                    self.stats["done"] += 1
                    self.last_summary = e.summary or e.result[:200]
                    print(f"[errands] {e.id} done: {self.last_summary[:120]}")
                    if self.on_done:
                        self.on_done(e)
                else:
                    self.stats["failed"] += 1
                    print(f"[errands] {e.id} failed: {(e.summary or e.result)[:120]}")
                    if self.on_fail:
                        self.on_fail(e)

    def _deliver_events(self) -> None:
        if self.deliver is None:
            return
        while True:
            with self._lock:
                if not self._events:
                    return
                text = self._events[0]
            try:
                ok = bool(self.deliver(text))
            except Exception as ex:
                print(f"[errands] deliver failed: {ex}")
                ok = False
            if not ok:
                return                                # the room is busy; try again next cycle
            with self._lock:
                self._events.pop(0)
