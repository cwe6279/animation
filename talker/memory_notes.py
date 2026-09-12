"""
memory_notes.py — an assistant's working memory: two markdown files next to face.json.

    notes.md   what she chose to remember, one line at a time, plus a short
               summary written when a session ends
    tasks.md   the ledger of work handed to the backend agent and its state

Both are plain markdown a person can open and edit. They are gitignored: they
hold your actual business. The brain never touches the filesystem; it writes a
{{note ...}} action and the handler here does the append. At launch the tail of
each file is filled into character.md through the {notes} and {tasks}
placeholders (see launch_facts.py), which is how she knows where she left off.

    nb = Notebook(face_dir);  nb.note("the board moved to Thursday")
    led = TaskLedger(face_dir); led.add("3f2a", "find three projectors under 300")
    led.set_state("3f2a", "done", summary="The Epson ... ")
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional

STATES = ("not started", "in progress", "done", "failed")


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now()


class Notebook:
    """notes.md: dated one-liners under a heading per day."""

    FILE = "notes.md"

    def __init__(self, face_dir: str, title: str = "Notes"):
        self.path = os.path.join(face_dir, self.FILE)
        self.title = title
        self._lock = threading.Lock()

    def _read(self) -> str:
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _append(self, block: str, now: datetime) -> None:
        """Append under today's `## date` heading, adding the heading if the file
        does not end in today's section."""
        day = f"## {now.strftime('%Y-%m-%d')}"
        with self._lock:
            text = self._read()
            parts = []
            if not text:
                parts.append(f"# {self.title}\n")
            last_heading = None
            for m in re.finditer(r"^## (\S+)\s*$", text, re.M):
                last_heading = m.group(0).strip()
            if last_heading != day:
                parts.append(f"\n{day}\n")
            parts.append(block.rstrip("\n") + "\n")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("".join(parts))

    def note(self, text: str, now: Optional[datetime] = None) -> str:
        """Write one line. Returns the line as written (for the log)."""
        text = " ".join(text.split())
        if not text:
            return ""
        now = _now(now)
        line = f"- {now.strftime('%H:%M')} — {text}"
        self._append(line, now)
        return line

    def session_summary(self, text: str, now: Optional[datetime] = None) -> None:
        text = text.strip()
        if not text:
            return
        now = _now(now)
        self._append(f"\n### Session summary {now.strftime('%H:%M')}\n{text}\n", now)

    def recent(self, max_chars: int = 3000) -> str:
        """The tail of the file for the prompt; the oldest lines fall off first."""
        text = self._read().strip()
        if not text:
            return "(no notes yet)"
        body = re.sub(r"^# .*\n", "", text, count=1).strip()   # the title is not a note
        if len(body) <= max_chars:
            return body
        cut = body[-max_chars:]
        nl = cut.find("\n")
        return "… (older notes omitted)\n" + (cut[nl + 1:] if nl >= 0 else cut)


class TaskLedger:
    """tasks.md: one entry per task, its state first so it scans at a glance.

        - [in progress] 3f2a · 2026-09-12 14:02 · research the supplier delays
        - [done] 1a2b · 2026-09-12 13:00 · find three projectors under 300
            The Epson EF-11 at 280 is the pick; two others are listed in the result.
    """

    FILE = "tasks.md"
    _LINE = re.compile(r"^- \[(not started|in progress|done|failed)\] (\S+) · (\S+ \S+) · (.*)$")

    def __init__(self, face_dir: str, title: str = "Tasks"):
        self.path = os.path.join(face_dir, self.FILE)
        self.title = title
        self._lock = threading.Lock()
        self.items: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        items: List[Dict] = []
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            return items
        for line in lines:
            m = self._LINE.match(line)
            if m:
                items.append({"state": m.group(1), "id": m.group(2), "when": m.group(3),
                              "task": m.group(4).strip(), "summary": ""})
            elif line.startswith("    ") and items:
                items[-1]["summary"] = (items[-1]["summary"] + "\n" + line[4:]).strip()
        return items

    def _save(self) -> None:
        out = [f"# {self.title}", ""]
        for it in self.items:
            out.append(f"- [{it['state']}] {it['id']} · {it['when']} · {it['task']}")
            for ln in (it["summary"] or "").splitlines():
                out.append(f"    {ln}")
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, self.path)

    def add(self, task_id: str, task: str, now: Optional[datetime] = None) -> Dict:
        task = " ".join(task.split())
        it = {"state": "not started", "id": task_id, "when": _now(now).strftime("%Y-%m-%d %H:%M"),
              "task": task, "summary": ""}
        with self._lock:
            self.items.append(it)
            self._save()
        return it

    def set_state(self, task_id: str, state: str, summary: Optional[str] = None) -> bool:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, not {state!r}")
        with self._lock:
            for it in self.items:
                if it["id"] == task_id:
                    it["state"] = state
                    if summary is not None:
                        it["summary"] = " ".join(summary.split())
                    self._save()
                    return True
        return False

    def get(self, task_id: str) -> Optional[Dict]:
        return next((it for it in self.items if it["id"] == task_id), None)

    def open_items(self) -> List[Dict]:
        return [it for it in self.items if it["state"] in ("not started", "in progress")]

    def render(self, max_chars: int = 2000) -> str:
        """For the prompt: every open task, then the most recent finished ones that fit."""
        if not self.items:
            return "(no tasks yet)"
        lines = []
        for it in self.items:
            s = f"- [{it['state']}] {it['id']} · {it['when']} · {it['task']}"
            if it["summary"]:
                s += f"\n    {it['summary']}"
            lines.append((it["state"] in ("not started", "in progress"), s))
        keep = [s for open_, s in lines if open_]
        budget = max_chars - sum(len(s) + 1 for s in keep)
        closed = [s for open_, s in lines if not open_]
        tail: List[str] = []
        for s in reversed(closed):                       # newest finished first
            if budget - len(s) - 1 < 0:
                break
            budget -= len(s) + 1
            tail.insert(0, s)
        omitted = len(closed) - len(tail)
        out = ([f"… ({omitted} older finished tasks omitted)"] if omitted else []) + tail + keep
        return "\n".join(out)
