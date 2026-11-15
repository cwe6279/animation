"""
talker/memtrace.py — opt-in memory growth report, for finding a slow leak on the Pi.

TALKER_MEMTRACE=1 starts tracemalloc once the warm-up is over and takes a
baseline snapshot, then every TALKER_MEMTRACE_MIN minutes (default 15) logs the
process RSS + swap and the allocation sites that grew the most since the baseline.
Tracing costs memory and CPU, so leave it off unless you are hunting a leak.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import tracemalloc

FRAMES = 4      # enough call stack to see who called into a library, cheap enough for a Pi Zero


def _proc_kb() -> str:
    try:
        with open("/proc/self/status") as f:
            vals = {k: v.split()[0] for k, v in (line.split(":", 1) for line in f) if k in ("VmRSS", "VmSwap")}
        return f"rss {int(vals['VmRSS']) // 1024} MB, swap {int(vals.get('VmSwap', 0)) // 1024} MB"
    except (OSError, KeyError, ValueError):
        return "rss unknown"


WATCH = ("SSLProtocol", "_SSLProtocolTransport", "SSLObject", "ClientSession", "ClientWebSocketResponse",
         "TCPConnector", "Task")


def _live_objects() -> None:
    """Counts of live network objects after a full collection, and who holds the TLS ones:
    a count that climbs with every reconnect is a connection nobody let go of."""
    found = gc.collect()
    by_type: dict = {}
    for o in gc.get_objects():
        name = type(o).__name__
        if name in WATCH:
            by_type.setdefault(name, []).append(o)
    print(f"[memtrace] live after gc ({found} unreachable freed): "
          + ", ".join(f"{n} {len(by_type.get(n, []))}" for n in WATCH))
    protos = by_type.get("SSLProtocol", [])
    if len(protos) > 2:
        holders: dict = {}
        old = protos[:-2]                        # the newest ones are the live connections
        for p in old:
            for r in gc.get_referrers(p):
                if r is protos or r is old:
                    continue
                desc = type(r).__name__
                if isinstance(r, dict):          # usually an object's __dict__: name its owner
                    owners = [type(x).__name__ for x in gc.get_referrers(r) if getattr(x, "__dict__", None) is r]
                    desc = f"dict of {owners[0]}" if owners else "dict"
                holders[desc] = holders.get(desc, 0) + 1
        print(f"[memtrace]   old SSLProtocol held by: {holders}")
        del old
    del by_type, protos


def _report(baseline, top: int) -> None:
    _live_objects()
    snap = tracemalloc.take_snapshot().filter_traces([
        tracemalloc.Filter(False, tracemalloc.__file__), tracemalloc.Filter(False, "<frozen importlib._bootstrap*>")])
    traced, peak = tracemalloc.get_traced_memory()
    print(f"[memtrace] {_proc_kb()}; python traced {traced / 2**20:.1f} MB (peak {peak / 2**20:.1f})")
    for st in snap.compare_to(baseline, "traceback")[:top]:
        if st.size_diff <= 0:
            break
        where = " <- ".join(f"{os.path.basename(fr.filename)}:{fr.lineno}" for fr in reversed(st.traceback))
        print(f"[memtrace]   {st.size_diff / 1024:+.0f} KB in {st.count_diff:+d} blocks at {where}")


def start_from_env() -> bool:
    """Starts the report thread (which starts tracing after warm-up) if TALKER_MEMTRACE is set."""
    if not os.environ.get("TALKER_MEMTRACE"):
        return False
    every = max(1.0, float(os.environ.get("TALKER_MEMTRACE_MIN", "15"))) * 60

    def run():
        # Tracing starts only after startup: tracing the library imports and model loads
        # swamped a Pi Zero. A leak is in what gets allocated later, so nothing is lost.
        time.sleep(min(every, 300))
        tracemalloc.start(FRAMES)
        baseline = tracemalloc.take_snapshot()
        print(f"[memtrace] baseline taken; {_proc_kb()}; reporting every {every / 60:.0f} min")
        while True:
            time.sleep(every)
            try:
                _report(baseline, top=10)
            except Exception as e:      # a diagnostic must never take the assistant down
                print(f"[memtrace] report failed: {e!r}")

    threading.Thread(target=run, name="memtrace", daemon=True).start()
    return True
