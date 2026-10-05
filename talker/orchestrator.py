"""
orchestrator.py — the character as a manager of agent work, not a messenger.

A request handed off with {{task ...}} becomes a goal. A planner (a fast model,
off the voice path) breaks it into ordered steps; each step goes to the backend
agent through ErrandRunner once the steps it depends on are done, with their
results passed in. The orchestrator then chases every step to the end:

    the agent is unreachable / timed out    -> retried quietly (RETRIES times)
    the agent stops to ask permission       -> if the person's own request covers it (auto
                                               mode), the step is resent with that approval
                                               spelled out; otherwise one question goes to
                                               the person, answered with {{approve g1}} /
                                               {{deny g1}} or the control page
    the agent fails outright                -> one more attempt told what went wrong, then
                                               a short report with whatever was gathered
    every step is done                      -> one announcement for the whole goal

With a protocol 2 agent (errands.py) the same happens without resending: an approved
action goes out with an "approvals" entry (the agent then never asks), a question the
request covers is answered "yes" by the character (by "clara") only when the targets in
the agent's pending action appear in the request, the owner's answer resumes the paused
task, and failures are handled by their error_kind.

The person hears from the character twice per goal at most: when it is handed off
and when it is finished, plus a question only when a decision is really theirs.

    orch = Orchestrator(runner, planner=make_claude_planner(), announce=runner.say_later)
    orch.start_goal("look up the Hilton Head forecast and text it to the owner at +1 917 ...")
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

RETRIES = 2                  # quiet retries for an unreachable / timed-out agent
REDO = 1                     # extra attempts for a step that failed outright, told why

# How a relay says "I stopped to ask" today: a failed task whose summary or result says so.
_ASKED = re.compile(r"needed an answer or a confirmation|\(yes/no\)|proceed\?|confirm(ation)? (is )?required|"
                    r"awaiting (your )?(approval|confirmation)", re.I)
_TRANSIENT = re.compile(r"not reachable|timed? ?out|relay restarted|connection (refused|reset)|temporar", re.I)


@dataclass
class Step:
    n: int
    do: str
    after: List[int] = field(default_factory=list)
    kind: str = "info"               # info | action
    approved: bool = False           # the person's request already covers this action
    state: str = "waiting"           # waiting | running | done | needs_input | failed | skipped
    errand_id: Optional[str] = None
    tries: int = 0
    redo: int = 0
    asked: int = 0                   # times the agent stopped to ask about this step
    summary: str = ""
    result: str = ""
    question: str = ""
    action: Optional[dict] = None    # protocol 2: the agent's exact pending tool call


@dataclass
class Goal:
    id: str
    text: str
    context: str = ""
    steps: List[Step] = field(default_factory=list)
    state: str = "planning"          # planning | running | needs_input | done | failed
    created: float = field(default_factory=time.time)
    reported: bool = False

    def step(self, n: int) -> Optional[Step]:
        return next((s for s in self.steps if s.n == n), None)

    def progress(self) -> str:
        return "; ".join(f"step {s.n} {s.state}" + (f": {s.summary[:120]}" if s.summary else "")
                         for s in self.steps)


def agent_asked(summary: str, result: str) -> bool:
    return bool(_ASKED.search(summary or "") or _ASKED.search((result or "")[-600:]))


_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _phones(text: str) -> set:
    return {re.sub(r"\D", "", m)[-10:] for m in _PHONE.findall(text or "") if len(re.sub(r"\D", "", m)) >= 10}


def targets_covered(action: Optional[dict], text: str) -> bool:
    """True when every phone number and email address in the agent's pending tool call also
    appears in what the person asked for. A call with no such target is covered by the
    step's approval alone (the planner only approves what the request explicitly asks)."""
    if not action:
        return True
    args = " ".join(str(v) for v in (action.get("args") or {}).values())
    phones, emails = _phones(args), {e.lower() for e in _EMAIL.findall(args)}
    asked_phones, asked_emails = _phones(text), {e.lower() for e in _EMAIL.findall(text or "")}
    return phones <= asked_phones and emails <= asked_emails


def approvals_for(text: str) -> list:
    """Protocol 2 approvals derived from an approved action step: the numbers it names."""
    return [{"tool": "send_sms", "args": {"to_number": "+1" + p if len(p) == 10 else p}} for p in sorted(_phones(text))]


def pending_question(summary: str, result: str) -> str:
    """The agent's own words about what it wants to do, for the person or for the approval."""
    tail = (result or "").strip()[-700:]
    return tail or (summary or "").strip()


# ─────────────────────────────────────────────────────
# PLANNER
# ─────────────────────────────────────────────────────
PLAN_PROMPT = """You plan work for a backend agent that can: {can}.
Break the request into the fewest ordered steps the agent can do one at a time. A step that needs an
earlier step's output lists it in "after"; its output is passed in automatically. Steps that do not
depend on each other have no "after" and run in parallel. Most requests are ONE step: only split when
a later step genuinely needs an earlier result (look something up, then send/write/book with it).

"kind" is "action" for anything that changes the world or reaches a person (sending a message or
email, booking, buying, writing to a calendar, deleting) and "info" for reading and looking up.
"approved" is true only for an action the request itself explicitly asks for with its target
(e.g. "text it to me at 917..." approves that one text). Never approve purchases or deleting.

Request: {goal}
Context: {context}

Reply with JSON only: {{"steps": [{{"n": 1, "do": "<one clear instruction with everything it needs>", "after": [], "kind": "info", "approved": false}}]}}"""


def parse_plan(text: str, goal: str) -> List[Step]:
    """The planner's JSON as Steps; one step for the whole goal if it is unusable."""
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        raw = json.loads(m.group(0))["steps"] if m else []
        steps = []
        for i, s in enumerate(raw, 1):
            do = " ".join(str(s.get("do", "")).split())
            if not do:
                continue
            n = int(s.get("n", i))
            after = [int(a) for a in (s.get("after") or []) if int(a) != n]
            kind = "action" if str(s.get("kind", "info")).lower() == "action" else "info"
            steps.append(Step(n=n, do=do, after=after, kind=kind, approved=bool(s.get("approved")) and kind == "action"))
        known = {s.n for s in steps}
        for s in steps:                                  # drop dependencies on steps that do not exist
            s.after = [a for a in s.after if a in known and a < s.n]
        if steps:
            return steps
    except Exception:
        pass
    return [Step(n=1, do=goal)]


def make_claude_planner(model: str = "claude-haiku-4-5", can: str = "") -> Callable[[str, str], List[Step]]:
    def plan(goal: str, context: str = "") -> List[Step]:
        try:
            from .brains.claude_chat import make_client
            resp = make_client().messages.create(
                model=model, max_tokens=700,
                messages=[{"role": "user", "content": PLAN_PROMPT.format(
                    can=can or "research, read and write", goal=goal, context=context or "(none)")}])
            text = "".join(getattr(b, "text", "") for b in resp.content)
        except Exception as e:
            print(f"[plan] planner unavailable ({e}); one step")
            text = ""
        return parse_plan(text, goal)
    return plan


# ─────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────
class Orchestrator:
    """Owns goals and their steps; ErrandRunner only carries single tasks to the agent."""

    def __init__(self, runner, planner: Callable[[str, str], List[Step]],
                 announce: Callable[[str], None], mode: str = "auto",
                 on_change: Optional[Callable[[Goal], None]] = None):
        self.runner = runner
        self.planner = planner
        self.announce = announce
        self.mode = mode                 # auto: the request's own approvals count; ask: every action asks
        self.on_change = on_change or (lambda g: None)
        self.goals: Dict[str, Goal] = {}
        self._by_errand: Dict[str, tuple] = {}
        self._lock = threading.RLock()

    # ── from the character ─────────────────────────────
    def start_goal(self, text: str, context: str = "", plan_now: bool = False) -> Goal:
        """Returns at once; planning (a model call) runs on its own thread."""
        g = Goal(id="g" + secrets.token_hex(2), text=" ".join(text.split()), context=context)
        with self._lock:
            self.goals[g.id] = g
        self.on_change(g)
        if plan_now:
            self._plan(g)
        else:
            threading.Thread(target=self._plan, args=(g,), daemon=True, name=f"plan-{g.id}").start()
        return g

    def decide(self, goal_id: str, approve: bool) -> str:
        """The person's answer to a pending question, from {{approve g1}} or the control page."""
        with self._lock:
            g = self.goals.get(goal_id) or self._only_waiting()
            if g is None:
                return "no goal is waiting for a decision"
            asking = [s for s in g.steps if s.state == "needs_input"]
            if not asking:
                return f"{g.id} is not waiting for a decision"
            for s in asking:
                if s.errand_id and self.runner.answer(s.errand_id, "yes" if approve else "no", by="owner"):
                    s.state = "running"               # the paused task resumes where it stopped
                    continue
                if approve:                           # a version 1 agent: resend, approved
                    s.approved, s.asked = True, 0
                    self._send(g, s, approval=True)
                else:
                    s.state, s.summary = "skipped", "the owner said no"
                    self._skip_dependents(g, s.n)
            g.state = "running"
        print(f"[plan] {g.id} {'approved' if approve else 'declined'} by the owner")
        self._advance(g)
        return f"{g.id} {'approved' if approve else 'declined'}"

    def _only_waiting(self) -> Optional[Goal]:
        waiting = [g for g in self.goals.values() if g.state == "needs_input"]
        return waiting[0] if len(waiting) == 1 else None

    # ── planning and dispatch ─────────────────────────
    def _plan(self, g: Goal) -> None:
        steps = self.planner(g.text, g.context) or [Step(n=1, do=g.text)]
        if self.mode == "ask":
            for s in steps:
                s.approved = False
        with self._lock:
            g.steps, g.state = steps, "running"
        print(f"[plan] {g.id}: " + " -> ".join(
            f"{s.n}{'(after ' + ','.join(map(str, s.after)) + ')' if s.after else ''}"
            f"{' ' + s.kind + (' approved' if s.approved else '') if s.kind == 'action' else ''}: {s.do[:70]}"
            for s in steps))
        self._advance(g)

    def _advance(self, g: Goal) -> None:
        with self._lock:
            for s in g.steps:
                if s.state == "waiting" and all((g.step(a) or Step(0, "", state="done")).state == "done" for a in s.after):
                    self._send(g, s)
            states = {s.state for s in g.steps}
            if states <= {"done", "skipped"} and not g.reported:
                g.state, g.reported = "done", True
                self._report_done(g)
            elif "failed" in states and not ({"running", "waiting", "needs_input"} & states) and not g.reported:
                g.state, g.reported = "failed", True
                self._report_failed(g)
        self.on_change(g)

    def _send(self, g: Goal, s: Step, approval: bool = False) -> None:
        parts = [s.do]
        for a in s.after:
            dep = g.step(a)
            if dep is not None and (dep.summary or dep.result):
                parts.append(f"Result of the earlier step ({dep.do[:80]}): {dep.result or dep.summary}"[:2500])
        if approval or (s.approved and s.kind == "action"):
            parts.append("APPROVED IN ADVANCE: the owner explicitly asked for exactly this. Do not stop to ask "
                         "for confirmation; go ahead and do it, then report what was done.")
            if s.question:
                parts.append(f"The action you proposed last time, which is approved: {s.question[-500:]}")
        if s.redo and s.summary:
            parts.append(f"A previous attempt failed: {s.summary[:300]} Try a different way.")
        approvals = approvals_for(f"{s.do} {g.text}") if (s.approved and s.kind == "action") else []
        e = self.runner.submit(" ".join(parts), context=g.context, approvals=approvals)
        s.errand_id, s.state, s.tries = e.id, "running", s.tries + 1
        self._by_errand[e.id] = (g.id, s.n)

    # ── from ErrandRunner (its poller thread) ─────────
    def owns(self, errand_id: str) -> bool:
        return errand_id in self._by_errand

    def on_errand_done(self, e) -> None:
        g, s = self._lookup(e.id)
        if s is None:
            return
        with self._lock:
            s.state, s.summary, s.result = "done", e.summary or e.result[:300], e.result
        self._advance(g)

    def on_errand_needs_input(self, e) -> None:
        """Protocol 2: the agent paused to ask. Answer it if the request covers it; else ask once."""
        g, s = self._lookup(e.id)
        if s is None:
            return
        with self._lock:
            s.summary, s.result = e.summary or s.summary, e.result or s.result
            s.question, s.action = e.question or pending_question(e.summary, e.result), e.action
            covered = (self.mode == "auto" and s.approved and e.options[:1] == ["yes"]
                       and targets_covered(e.action, f"{s.do} {g.text} {g.context}"))
            if covered and self.runner.answer(e.id, "yes", by="clara"):
                print(f"[plan] {g.id} step {s.n}: the agent asked; the request covers it, answered yes")
                s.state = "running"
            else:
                s.state, g.state = "needs_input", "needs_input"
                self.announce(f'Goal {g.id} "{g.text}" needs a decision from the owner. The agent asks: '
                              f'{s.question[-400:]} Ask in a few words; when they answer, write '
                              f'{{{{approve {g.id}}}}} or {{{{deny {g.id}}}}}'
                              + (" (or pass on their own words if it is an open question)." if not e.options else "."))
        self.on_change(g)

    def on_errand_failed(self, e) -> None:
        g, s = self._lookup(e.id)
        if s is None:
            return
        summary, result = e.summary or "", e.result or ""
        kind = getattr(e, "error_kind", "") or ""
        with self._lock:
            s.summary, s.result = summary, result
            if kind:                                  # protocol 2 says why; no guessing from wording
                if kind == "transient" and s.tries <= RETRIES:
                    print(f"[plan] {g.id} step {s.n}: transient; retrying")
                    self._send(g, s)
                elif kind == "tool_error" and s.redo < REDO:
                    s.redo += 1
                    print(f"[plan] {g.id} step {s.n}: tool error ({summary[:60]}); one more try, told why")
                    self._send(g, s)
                else:
                    if kind == "needs_input_expired":
                        s.question = getattr(e, "question", "") or s.question
                        s.summary = ("nobody answered the agent's question in time"
                                     + (f": {s.question[:200]}" if s.question else ""))
                    s.state = "failed"
                    self._skip_dependents(g, s.n)
            elif agent_asked(summary, result):
                s.question = pending_question(summary, result)
                s.asked += 1
                if s.approved and s.asked == 1 and self.mode == "auto":
                    print(f"[plan] {g.id} step {s.n}: the agent asked; the request covers it, resending approved")
                    self._send(g, s, approval=True)
                elif s.approved and s.asked > 1:
                    # Approved in the task and it still asks: it confirms at its own end; say so once.
                    s.state = "failed"
                    s.summary = ("the agent still asks for confirmation at its own end even when approved; "
                                 "it has to be allowed there (its settings or its own screen)")
                else:
                    s.state, g.state = "needs_input", "needs_input"
                    self.announce(f'Goal {g.id} "{g.text}" needs a decision from the owner: the agent wants to: '
                                  f'{s.question[-400:]} Ask in a few words; when they answer, write '
                                  f'{{{{approve {g.id}}}}} or {{{{deny {g.id}}}}}.')
            elif _TRANSIENT.search(summary) and s.tries <= RETRIES:
                print(f"[plan] {g.id} step {s.n}: transient ({summary[:60]}); retrying")
                self._send(g, s)
            elif s.redo < REDO:
                s.redo += 1
                print(f"[plan] {g.id} step {s.n}: failed ({summary[:60]}); one more try, told why")
                self._send(g, s)
            else:
                s.state = "failed"
                self._skip_dependents(g, s.n)
        self._advance(g)

    def _lookup(self, errand_id: str):
        with self._lock:
            gid, n = self._by_errand.get(errand_id, (None, None))
            g = self.goals.get(gid) if gid else None
            return g, (g.step(n) if g else None)

    def _skip_dependents(self, g: Goal, n: int) -> None:
        for s in g.steps:
            if n in s.after and s.state == "waiting":
                s.state, s.summary = "skipped", f"needed step {n}"
                self._skip_dependents(g, s.n)

    # ── reporting (one announcement per goal) ─────────
    def _report_done(self, g: Goal) -> None:
        results = " ".join(f"({s.n}) {s.summary or s.result[:400]}" for s in g.steps if s.state == "done")
        skipped = [s for s in g.steps if s.state == "skipped"]
        tail = f" Not done: {'; '.join(s.do[:80] + ' (' + s.summary + ')' for s in skipped)}." if skipped else ""
        print(f"[plan] {g.id} done")
        self.announce(f'Goal {g.id} "{g.text}" finished. Results: {results}{tail}')

    def _report_failed(self, g: Goal) -> None:
        got = " ".join(f"({s.n}) {s.summary or s.result[:300]}" for s in g.steps if s.state == "done")
        bad = "; ".join(f"step {s.n} ({s.do[:80]}): {s.summary[:200]}" for s in g.steps if s.state == "failed")
        print(f"[plan] {g.id} failed: {bad[:120]}")
        self.announce(f'Goal {g.id} "{g.text}" could not be finished. What went wrong: {bad}.'
                      + (f" What was gathered anyway: {got}" if got else "")
                      + " Tell the person the useful part first, then the problem in one sentence.")

    # ── for the control page and the ledger ───────────
    def snapshot(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            goals = sorted(self.goals.values(), key=lambda g: g.created, reverse=True)[:limit]
            return [{"id": g.id, "text": g.text, "state": g.state, "created": g.created,
                     "steps": [{"n": s.n, "do": s.do, "after": s.after, "kind": s.kind, "approved": s.approved,
                                "state": s.state, "tries": s.tries, "summary": s.summary, "question": s.question[-400:],
                                "action": s.action}
                               for s in g.steps]} for g in goals]
