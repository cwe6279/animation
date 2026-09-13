#!/usr/bin/env python3
"""
agent_relay.py — turn a command-line agent harness into the small task service
that talker/errands.py polls. Runs on the machine that has the harness; needs
nothing beyond Python 3.

    RELAY_COMMAND="claude -p" python tools/agent_relay.py          # port 8030

The prompt goes to the command on stdin and its stdout is the result, so any
harness with a non-interactive mode works (Claude Code: `claude -p`; for one
that has no such mode, a two-line wrapper script that feeds it and prints).

Two agents talking to each other: RELAY_ROUNDS=2 runs the command a second time
with the first answer and asks it to critique and improve it before the result
is returned. The character never sees the rounds; it gets a summary.

API (JSON):
    POST /tasks        {"id": "3f2a", "task": "...", "context": "...", "from": "clara"}  -> {"id": "3f2a"}
    GET  /tasks/{id}   -> {"id", "state": "queued|running|done|failed", "summary", "result", "created", "updated"}
    GET  /tasks        -> [ ... ]
    GET  /health       -> {"ok": true, "running": n}
    GET  /capabilities -> {"can": [...]}   from RELAY_CAN ("search the web; read the calendar; ...")

Settings (environment or flags):
    RELAY_PORT      8030            RELAY_HOST     0.0.0.0
    RELAY_COMMAND   "claude -p"     the harness, reads stdin, writes stdout
    RELAY_ROUNDS    1               2 = a second critique-and-revise pass
    RELAY_TIMEOUT   900             seconds per command run
    RELAY_STATE     relay_tasks.json   where tasks are kept across restarts
    RELAY_WORKERS   2               tasks run at the same time
    RELAY_CAN       ""              what the harness can do, semicolon-separated; told to the character

This binds to the network with no authentication: it is for a trusted LAN.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TASK_PROMPT = """You are working on behalf of a voice assistant. Do this task thoroughly and give the
finished result, written for a reader, with no preamble and no questions back.

Task: {task}
{context_block}"""

REVIEW_PROMPT = """A colleague was given this task and produced the answer below. Check it critically:
find mistakes, gaps and anything unverified, fix them, and return only the final improved
answer, written for a reader, with no commentary about the review.

Task: {task}
{context_block}
Colleague's answer:
{result}"""

SUMMARY_PROMPT = """Below is the finished result of a task. Write what an assistant should say out loud to
report it: two or three plain spoken sentences with the conclusion first, no lists, no
markdown, no preamble. If the result is a failure or empty, say so plainly.

Task: {task}

Result:
{result}"""


class Relay:
    def __init__(self, command: str, rounds: int, timeout: float, state_path: str, workers: int,
                 can: str = ""):
        self.command = shlex.split(command)
        self.can = [c.strip() for c in can.split(";") if c.strip()]
        self.rounds = max(1, rounds)
        self.timeout = timeout
        self.state_path = state_path
        self.tasks: dict = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max(1, workers))
        self._load()

    # ── persistence ────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                self.tasks = json.load(f)
        except (FileNotFoundError, ValueError):
            self.tasks = {}
        for t in self.tasks.values():           # a run interrupted by a restart is a failure, said so
            if t["state"] in ("queued", "running"):
                t["state"], t["summary"] = "failed", "The relay restarted before this finished."

    def _save(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.tasks, f, indent=1)
        os.replace(tmp, self.state_path)

    # ── the work ───────────────────────────────────────
    def run_agent(self, prompt: str) -> str:
        p = subprocess.run(self.command, input=prompt, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0 and not p.stdout.strip():
            raise RuntimeError(f"{self.command[0]} exited {p.returncode}: {p.stderr.strip()[:400]}")
        return p.stdout.strip()

    def submit(self, task: str, context: str = "", task_id: str = "", sender: str = "") -> str:
        task_id = task_id or f"{int(time.time() * 1000):x}"[-6:]
        entry = {"id": task_id, "task": task, "context": context, "from": sender, "state": "queued",
                 "summary": "", "result": "", "created": time.time(), "updated": time.time()}
        with self._lock:
            self.tasks[task_id] = entry
            self._save()
        threading.Thread(target=self._work, args=(task_id,), daemon=True, name=f"task-{task_id}").start()
        return task_id

    def _set(self, task_id: str, **fields) -> None:
        with self._lock:
            self.tasks[task_id].update(fields, updated=time.time())
            self._save()

    def _work(self, task_id: str) -> None:
        t = self.tasks[task_id]
        ctx = f"\nContext from the conversation: {t['context']}\n" if t["context"] else ""
        with self._sem:
            self._set(task_id, state="running")
            print(f"[relay] {task_id} running: {t['task'][:100]}")
            try:
                result = self.run_agent(TASK_PROMPT.format(task=t["task"], context_block=ctx))
                for _ in range(self.rounds - 1):
                    print(f"[relay] {task_id} review round")
                    result = self.run_agent(REVIEW_PROMPT.format(task=t["task"], context_block=ctx, result=result))
                summary = self.run_agent(SUMMARY_PROMPT.format(task=t["task"], result=result[:12000]))
                self._set(task_id, state="done", result=result, summary=summary)
                print(f"[relay] {task_id} done: {summary[:120]}")
            except Exception as e:
                self._set(task_id, state="failed", summary=f"The task failed: {e}")
                print(f"[relay] {task_id} failed: {e}")


def make_handler(relay: Relay):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            if path == "/capabilities":
                return self._json(200, {"can": relay.can})
            if path == "/health":
                running = sum(1 for t in relay.tasks.values() if t["state"] == "running")
                return self._json(200, {"ok": True, "running": running})
            if path == "/tasks":
                return self._json(200, sorted(relay.tasks.values(), key=lambda t: t["created"]))
            if path.startswith("/tasks/"):
                t = relay.tasks.get(path[len("/tasks/"):])
                return self._json(200, t) if t else self._json(404, {"error": "no such task"})
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/tasks":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            task = str(body.get("task") or "").strip()
            if not task:
                return self._json(400, {"error": "task is required"})
            tid = relay.submit(task, str(body.get("context") or ""), str(body.get("id") or ""),
                               str(body.get("from") or ""))
            self._json(200, {"id": tid})

        def log_message(self, fmt, *args) -> None:   # quiet; the relay prints what matters
            pass
    return Handler


def main() -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=int(env("RELAY_PORT", "8030")))
    ap.add_argument("--host", default=env("RELAY_HOST", "0.0.0.0"))
    ap.add_argument("--command", default=env("RELAY_COMMAND", "claude -p"),
                    help="the agent harness; the prompt is piped to its stdin, its stdout is the result")
    ap.add_argument("--rounds", type=int, default=int(env("RELAY_ROUNDS", "1")),
                    help="2 = a second agent pass that critiques and improves the first")
    ap.add_argument("--timeout", type=float, default=float(env("RELAY_TIMEOUT", "900")))
    ap.add_argument("--state", default=env("RELAY_STATE", "relay_tasks.json"))
    ap.add_argument("--workers", type=int, default=int(env("RELAY_WORKERS", "2")))
    ap.add_argument("--can", default=env("RELAY_CAN", ""),
                    help="what the harness can do, semicolon-separated; served at /capabilities")
    a = ap.parse_args()
    relay = Relay(a.command, a.rounds, a.timeout, a.state, a.workers, can=a.can)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(relay))
    srv.daemon_threads = True
    print(f"[relay] listening on http://{a.host}:{a.port}  command={a.command!r} rounds={a.rounds} "
          f"state={a.state}  ({len(relay.tasks)} tasks on file)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
