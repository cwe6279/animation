"""
talker/session_log.py — mirror everything printed to a per-session log file.

    from .session_log import start_session_log
    start_session_log("voice")          # -> logs/voice-20260907-181530.log

Keeps the terminal output unchanged; the copy in logs/ (gitignored) lets a
session be reviewed afterwards. The last 20 logs are kept.
"""

from __future__ import annotations

import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
LOG_DIR = os.path.join(HERE, "logs")


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        try:
            self.fh.write(data)
            self.fh.flush()
        except Exception:
            pass

    def flush(self):
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


def start_session_log(name: str = "session", keep: int = 20) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log")
    fh = open(path, "a", encoding="utf-8")
    fh.write(f"# {name} session started {time.strftime('%Y-%m-%d %H:%M:%S')}  argv: {' '.join(sys.argv[1:])}\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    for old in sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass
    return path


def latest_log(name: str = "") -> str | None:
    files = sorted(glob.glob(os.path.join(LOG_DIR, f"{name}*.log")))
    return files[-1] if files else None
