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

Protocol 2 (the agent says so in /health: "protocol": 2, "features": [...]) adds:
an "approvals" list on POST /tasks, a "needs_input" state carrying question /
options / action, POST {url}/tasks/{id}/answer {"answer", "by"} to resume it, and
an "error_kind" on failures. A shared token (AGENT_RELAY_TOKEN) goes in the
X-Relay-Token header. Version 1 agents keep working: the extras are just absent.

Polling, not callbacks: it works through any firewall and needs no open port on
the character's machine. A backend that is down is retried every cycle and said
once in the log, never raised.

    runner = ErrandRunner("http://agentbox:8030", on_done=lambda e: ...)
    runner.start()
    runner.submit("find three 4K projectors under 300")     # from any thread, ~1 µs
    runner.say_later("Task 3f2a finished: ...")           # tried each cycle until deliver() takes it

The address can be empty (no agent configured yet) and changed while running
with set_url(): tasks submitted meanwhile wait in the outbox. AgentControl is
what the control page talks to: status, a /health test, and a saved address.
"""

from __future__ import annotations

import json
import os
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
    approvals: List[dict] = field(default_factory=list)   # protocol 2: tool calls approved in advance
    question: str = ""                                    # protocol 2: needs_input
    options: List[str] = field(default_factory=list)
    action: Optional[dict] = None
    error_kind: str = ""                                  # protocol 2: transient | tool_error | refused | needs_input_expired
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)


def _http(url: str, method: str = "GET", body: Optional[dict] = None, timeout: float = 5.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    token = os.environ.get("AGENT_RELAY_TOKEN", "").strip()
    if token:
        headers["X-Relay-Token"] = token
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
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
                 on_needs_input: Optional[Callable[[Errand], None]] = None,
                 deliver: Optional[Callable[[str], bool]] = None,
                 fetch: Callable[..., dict] = _http, sender: str = "talker",
                 timeout: float = 10.0):
        self.url = (url or "").strip().rstrip("/")
        self.poll_s = poll_s
        # Every request to the agent waits this long. Five seconds is plenty on a LAN and
        # turns a working agent into a false "unreachable" over hotel wifi or a relayed
        # tailnet hop, which is where she is when the backend is furthest away.
        self.timeout = timeout
        self.on_done = on_done
        self.on_fail = on_fail
        self.on_started = on_started
        self.on_needs_input = on_needs_input
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
        self.protocol = 1                # what the agent's /health says; 2 enables answers
        self.features: List[str] = []
        self._answers: List[tuple] = []
        self.stats = {"submitted": 0, "done": 0, "failed": 0, "cycles": 0, "errors": 0}
        self.last_summary = ""

    # ── any thread ─────────────────────────────────────
    def submit(self, task: str, context: str = "", approvals: Optional[List[dict]] = None) -> Errand:
        """Queue a task; returns at once. The poller posts it on its next pass."""
        e = Errand(id=secrets.token_hex(2), task=" ".join(task.split()), context=context.strip(),
                   approvals=list(approvals or []))
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
            return [e for e in self.errands.values() if e.state in ("queued", "running", "needs_input", "answering")]

    def answer(self, errand_id: str, answer: str, by: str = "owner") -> bool:
        """Protocol 2: answer a task that is waiting for input; sent by the poller thread.
        False when this agent cannot take answers or the task is not waiting."""
        with self._lock:
            e = self.errands.get(errand_id)
            if e is None or e.state != "needs_input" or "answer" not in self.features:
                return False
            e.state = "answering"
            self._answers.append((e, answer, by))
        self._wake.set()
        return True

    def set_url(self, url: str) -> None:
        """Point at another backend (or none) while running. Queued tasks go to the new one;
        tasks already running on the old one keep their ids and are polled at the new address,
        which is right when the same relay just moved and wrong only if it is a different relay."""
        url = (url or "").strip().rstrip("/")
        if url != self.url:
            print(f"[errands] backend address {'set to ' + url if url else 'cleared'}")
            self.url = url
            self.reachable = None
            self._wake.set()

    def health(self, url: Optional[str] = None) -> Dict[str, object]:
        """GET {url}/health once; never raises. For the control page's Test button."""
        url = (url if url is not None else self.url).strip().rstrip("/")
        if not url:
            return {"ok": False, "detail": "no address set"}
        t0 = time.perf_counter()
        try:
            resp = self.fetch(f"{url}/health", timeout=self.timeout)
        except Exception as ex:
            return {"ok": False, "detail": str(ex)[:200], "ms": round((time.perf_counter() - t0) * 1000)}
        if url == self.url:
            self.protocol = int(resp.get("protocol") or 1)
            self.features = [str(f) for f in (resp.get("features") or [])]
        return {"ok": bool(resp.get("ok", True)), "running": resp.get("running"),
                "needs_input": resp.get("needs_input"), "protocol": int(resp.get("protocol") or 1),
                "features": resp.get("features") or [],
                "detail": "reachable", "ms": round((time.perf_counter() - t0) * 1000)}

    def snapshot(self, limit: int = 30) -> List[Dict[str, object]]:
        """Newest first, for the control page."""
        with self._lock:
            items = sorted(self.errands.values(), key=lambda e: e.created, reverse=True)[:limit]
        return [{"id": e.id, "task": e.task, "state": e.state, "summary": e.summary,
                 "result": e.result[:2000], "question": e.question, "error_kind": e.error_kind,
                 "created": e.created, "updated": e.updated} for e in items]

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
        if self.url:                                  # no address yet: tasks wait in the outbox
            if self.reachable is None or self.stats["cycles"] % 60 == 1:
                self.health()                         # learn (and now and then re-learn) the protocol
            self._post_outbox()
            self._send_answers()
            self._poll_running()
        self._deliver_events()

    def _say_reachability(self, ok: bool, err: str = "") -> None:
        if ok != self.reachable:
            print(f"[errands] backend at {self.url} {'reachable' if ok else 'unreachable, will retry'}"
                  + (f": {err}" if err else ""))
            # Away from home the agent is a network away, and a handed-off task that
            # simply queues looks like it was ignored. Say it once per change, and only
            # when something is actually waiting to go.
            with self._lock:
                waiting = len(self._outbox)
            what = "task" if waiting == 1 else f"{waiting} tasks"
            if waiting and not ok:
                self.say_later(
                    f"The agent cannot be reached right now, so the {what} you handed over has "
                    f"not gone yet. Say so briefly and that you will send it the moment the "
                    f"connection is back.")
            elif waiting and self.reachable is False:   # only if we said it was down
                self.say_later(
                    f"The agent can be reached again and the waiting {what} is going over now. "
                    f"Say so in a few words.")
        self.reachable = ok

    def _post_outbox(self) -> None:
        with self._lock:
            todo = list(self._outbox)
        for e in todo:
            try:
                body = {"id": e.id, "task": e.task, "context": e.context, "from": self.sender}
                if e.approvals and "approvals" in self.features:
                    body["approvals"] = e.approvals
                resp = self.fetch(f"{self.url}/tasks", "POST", body, timeout=self.timeout)
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

    def _send_answers(self) -> None:
        with self._lock:
            todo, self._answers = self._answers, []
        for i, (e, answer, by) in enumerate(todo):
            try:
                self.fetch(f"{self.url}/tasks/{e.remote_id}/answer", "POST",
                           {"answer": answer, "by": by}, timeout=self.timeout)
            except Exception as ex:
                if "409" in str(ex):                  # no longer waiting (expired / answered): the poll will say
                    e.state = "running"
                    continue
                self._say_reachability(False, str(ex))
                with self._lock:                      # keep this and the rest for the next cycle
                    self._answers = todo[i:] + self._answers
                e.state = "needs_input"
                return
            self._say_reachability(True)
            e.state, e.updated, e.question, e.action = "running", time.time(), "", None
            print(f"[errands] {e.id} answered {answer!r} (by {by})")

    def _poll_running(self) -> None:
        for e in self.pending():
            if e.state not in ("running", "needs_input"):
                continue
            try:
                resp = self.fetch(f"{self.url}/tasks/{e.remote_id}", timeout=self.timeout)
            except Exception as ex:
                self._say_reachability(False, str(ex))
                return
            self._say_reachability(True)
            state = str(resp.get("state", "running")).lower()
            if state == "needs_input" and e.state != "needs_input":
                e.state, e.updated = "needs_input", time.time()
                e.summary = str(resp.get("summary") or "").strip()
                e.result = str(resp.get("result") or "").strip()
                e.question = str(resp.get("question") or "").strip()
                e.options = [str(o) for o in (resp.get("options") or [])]
                e.action = resp.get("action") if isinstance(resp.get("action"), dict) else None
                print(f"[errands] {e.id} needs input: {e.question[:120]}")
                if self.on_needs_input:
                    self.on_needs_input(e)
                continue
            if state == "running" and e.state == "needs_input":
                e.state = "running"                       # answered elsewhere (the agent's own screen)
            if state in ("done", "failed"):
                e.state, e.updated = state, time.time()
                e.summary = str(resp.get("summary") or "").strip()
                e.result = str(resp.get("result") or "").strip()
                e.error_kind = str(resp.get("error_kind") or "").strip()
                e.question = str(resp.get("question") or e.question or "").strip()
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


class AgentControl:
    """What the control page's Agent tab drives: the runner's address, a reachability
    test, the task list, and saving the address to settings.json for the next start."""

    def __init__(self, runner: ErrandRunner, can: str = "", source: str = "",
                 save: Optional[Callable[[str], None]] = None, orch=None):
        self.runner = runner
        self.orch = orch              # orchestrator.Orchestrator: goals, steps, decisions, mode
        self.can = can                # what face.json says the agent can do
        self.source = source          # where the address came from at startup: flag | settings | env | none
        self._save = save

    def info(self) -> Dict[str, object]:
        r = self.runner
        return {"url": r.url, "source": self.source, "reachable": r.reachable, "poll_s": r.poll_s,
                "protocol": r.protocol, "features": r.features,
                "can": self.can, "stats": dict(r.stats), "open": len(r.pending()),
                "tasks": r.snapshot(), "mode": self.orch.mode if self.orch else None,
                "goals": self.orch.snapshot() if self.orch else []}

    def decide(self, goal_id: str, approve: bool) -> Dict[str, object]:
        if self.orch is None:
            return {"error": "no orchestrator in this run"}
        return {"message": self.orch.decide(goal_id, approve)}

    def set_mode(self, mode: str) -> Dict[str, object]:
        if self.orch is None or mode not in ("auto", "ask"):
            return {"error": "mode must be auto or ask"}
        self.orch.mode = mode
        return {"mode": mode, "message": f"errand mode: {mode} (this run; --errand-mode sets it at startup)"}

    def test(self, url: str = "") -> Dict[str, object]:
        return self.runner.health(url or None)

    def set_url(self, url: str, save: bool = True) -> Dict[str, object]:
        url = (url or "").strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            return {"error": "the address must start with http:// or https://"}
        self.runner.set_url(url)
        saved = False
        if save and self._save is not None:
            self._save(url)
            saved = True
        if self.source == "flag" and saved:
            note = "saved; note that --agent-url on the command line still wins at the next start"
        else:
            note = "saved for the next start" if saved else "applied for this run only"
        return {"url": self.runner.url, "message": note}
