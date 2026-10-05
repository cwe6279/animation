"""
incidents.py — a running record of what went wrong, for review and self-improvement.

Each incident is one JSON line in logs/incidents.jsonl (gitignored with the rest
of logs/): when, what kind, and the details that would help fix it. Kinds so far:

    net_offline / net_online   the internet went and came back (down_s)
    slow_turn                  a reply took longer than SLOW_TURN_MS to start (the timing line)
    goal_failed                an agent goal could not be finished (goal, why)
    stt_error                  the recognizer reported an error

The file is the raw material for a periodic self-review: what fails, how often,
and which setting or prompt change would help.

    from . import incidents
    incidents.record("slow_turn", first_audio_ms=7612, line="you stopped -> ...")
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "logs", "incidents.jsonl")
SLOW_TURN_MS = 5000
_lock = threading.Lock()


def record(kind: str, path: str = "", **details: Any) -> Dict[str, Any]:
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **details}
    target = path or PATH
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with _lock, open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass                                             # a full disk must not take the character down
    return entry


def recent(limit: int = 200, path: str = "") -> List[Dict[str, Any]]:
    try:
        with open(path or PATH, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out
