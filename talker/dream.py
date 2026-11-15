"""
dream.py — a nightly review of the character's own logs, written up as proposed improvements.

When the character has been dormant for a few hours and has not dreamed today, a strong
model (Claude Opus 5.5 by default; Claude Fable 5.1 with --dream-model) reads a digest
of everything since the last dream and writes dreams/<date>.md:

    what happened, what went well, problems (with times and quotes as evidence),
    proposed improvements (change, why, evidence, risk, kind: setting / prompt / code /
    hardware / agent), and questions for the owner.

Nothing is changed automatically: the report is for the owner to read and act on. The
digest is built locally and kept small (noise collapsed, long sessions trimmed), and the
call has a wall-clock limit (an hour) and an output cap, so one dream costs well under
a dollar on Claude Opus 5.5.

    python -m talker.dream --now          # dream once, now
    DreamScheduler(...).start()           # inside voice_loop: dreams when idle
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from collections import Counter
from typing import Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(HERE, "logs")
DREAMS = os.path.join(HERE, "dreams")
STATE = os.path.join(DREAMS, "state.json")

DEFAULT_MODEL = "claude-opus-5-5"
SESSION_CHARS = 40_000          # per session log, head and tail kept
DIGEST_CHARS = 350_000          # whole digest (~90k tokens)
MAX_OUTPUT = 24_000
TIME_LIMIT_S = 3600

# Lines worth reading; everything else (partial transcripts, level meters, library noise) is dropped.
KEEP = re.compile(r"\[(you|bot|event|task|plan|errands|error|net|brain|start|mode|turn|stt|calibration|"
                  r"speech|note|memory|vision|web|cues|g2p|audio)\]|Traceback|Error|error:")
DROP = re.compile(r"\[(hearing|level)|RTKit|INFO |WARN |^\s*$")
STAMP = re.compile(r"^\d\d:\d\d:\d\d ")

DREAM_PROMPT = """You are reviewing the logs of Clara, a voice assistant that runs on a Raspberry Pi Zero 2 W
(512 MB): a USB mic, a Bluetooth speaker, a camera, cloud speech recognition (ElevenLabs Scribe),
a fast Claude model for conversation, an ElevenLabs voice, and a backend agent she hands longer
tasks to. Her owner reads your report the next morning and decides what to change.

Write a report in Markdown with exactly these sections:

## Summary
Three or four sentences: how the period went, the one thing most worth fixing.

## What went well
Short bullets, with evidence.

## Problems
For each: what happened, when (timestamps from the logs), how often, a short quote as evidence,
and the likely cause. Most important first. Include slow turns, mishearings, wrong or awkward
replies, failed or stuck tasks, hardware and network trouble, crashes.

## Proposed improvements
A numbered list, most valuable first. For each: the change, why (link it to the problems), the
evidence, the risk, the kind (setting / prompt / code / hardware / agent), and whether it is safe
to apply without the owner (only settings within normal ranges ever are). Be concrete: name the
setting and value, the instruction text, or the file and function.

## Follow-up on earlier dreams
For proposals in the previous reports included below: which look done, which still apply.

## Questions for the owner
Only what you genuinely cannot tell from the logs.

Be specific and brief. Do not invent events that are not in the logs. Do not repeat secrets,
phone numbers or email addresses even if they appear in the logs."""


# ── digest ────────────────────────────────────────────────────────────────────
def condense(lines: List[str]) -> List[str]:
    """Keep the useful lines and collapse runs of the same message (ignoring the time stamp)."""
    out: List[str] = []
    last_body, count = None, 0
    for raw in lines:
        line = raw.rstrip("\n")
        if DROP.search(line) or not KEEP.search(line):
            continue
        body = STAMP.sub("", line)
        if body == last_body:
            count += 1
            continue
        if count:
            out[-1] += f"   (repeated x{count + 1})"
        out.append(line)
        last_body, count = body, 0
    if count:
        out[-1] += f"   (repeated x{count + 1})"
    return out


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 5], text[-limit * 3 // 5:]
    return f"{head}\n... [{len(text) - limit} characters of this session left out] ...\n{tail}"


def turn_stats(lines: List[str]) -> Dict[str, float]:
    t, k, a = [], [], []
    for line in lines:
        m = re.search(r"transcript (\d+) ms -> first token (\d+|\?) ms.* first audio (\d+) ms", line)
        if m:
            t.append(int(m.group(1))); a.append(int(m.group(3)))
            if m.group(2) != "?":
                k.append(int(m.group(2)))
    if not a:
        return {}
    avg = lambda xs: round(sum(xs) / len(xs)) if xs else None
    return {"turns": len(a), "avg_transcript_ms": avg(t), "avg_first_token_ms": avg(k),
            "avg_first_audio_ms": avg(a), "slow_turns_over_5s": sum(1 for x in a if x > 5000)}


def _read(path: str, tail_chars: int = 0) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            s = f.read()
        return s[-tail_chars:] if tail_chars else s
    except OSError:
        return ""


def build_digest(since: float, face_dir: str = "", extra_files: Optional[Dict[str, str]] = None,
                 logs_dir: str = LOGS, dreams_dir: str = DREAMS) -> Dict[str, object]:
    """Everything since `since` (epoch seconds), condensed. Returns {'text', 'sessions', 'chars'}."""
    parts: List[str] = []
    sessions = 0
    for path in sorted(glob.glob(os.path.join(logs_dir, "voice-*.log"))):
        if os.path.getmtime(path) < since:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.readlines()
        kept = condense(raw)
        stats = turn_stats(kept)
        sessions += 1
        header = f"### Session log {os.path.basename(path)}" + (f"  (turn timing: {json.dumps(stats)})" if stats else "")
        parts.append(header + "\n" + trim("\n".join(kept), SESSION_CHARS))
    inc_path = os.path.join(logs_dir, "incidents.jsonl")
    incidents = []
    for line in _read(inc_path).splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if time.mktime(time.strptime(row.get("time", "1970-01-01T00:00:00"), "%Y-%m-%dT%H:%M:%S")) >= since:
            incidents.append(row)
    if incidents:
        kinds = Counter(r["kind"] for r in incidents)
        parts.append("### Incidents\nCounts: " + json.dumps(kinds) + "\n" +
                     "\n".join(json.dumps(r) for r in incidents[-150:]))
    if face_dir:
        for name, chars in (("tasks.md", 5000), ("notes.md", 4000)):
            text = _read(os.path.join(face_dir, name), chars)
            if text:
                parts.append(f"### {name} (latest)\n{text}")
        face_json = _read(os.path.join(face_dir, "face.json"))
        if face_json:
            try:
                d = json.loads(face_json)
                keep = {k: d[k] for k in ("name", "models", "tts", "tts_model", "wake_words", "errands") if k in d}
                parts.append("### face.json (selected)\n" + json.dumps(keep))
            except ValueError:
                pass
    for label, path in (extra_files or {}).items():
        text = _read(path, 4000)
        if text:
            parts.append(f"### {label}\n{text}")
    previous = sorted(glob.glob(os.path.join(dreams_dir, "20*.md")))[-2:]
    for p in previous:
        parts.append(f"### Earlier dream {os.path.basename(p)}\n" + trim(_read(p), 12_000))
    text = trim("\n\n".join(parts), DIGEST_CHARS)
    return {"text": text, "sessions": sessions, "chars": len(text), "incidents": len(incidents)}


# ── the dream ─────────────────────────────────────────────────────────────────
def load_state(path: str = STATE) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state: Dict[str, object], path: str = STATE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def review(digest: str, model: str = DEFAULT_MODEL, client=None, time_limit_s: float = TIME_LIMIT_S) -> Dict[str, object]:
    """One long call; returns {'text', 'stop_reason', 'input_tokens', 'output_tokens'}."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    client = client.with_options(timeout=time_limit_s, max_retries=1) if hasattr(client, "with_options") else client
    kwargs = dict(model=model, max_tokens=MAX_OUTPUT, system=DREAM_PROMPT,
                  output_config={"effort": "high"},
                  messages=[{"role": "user", "content": "Logs and context since the last review:\n\n" + digest}])
    if model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    with client.beta.messages.stream(**kwargs) as stream:
        msg = stream.get_final_message()
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
    usage = getattr(msg, "usage", None)
    return {"text": text, "stop_reason": getattr(msg, "stop_reason", ""),
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0}


def dream(face_dir: str = "", model: str = DEFAULT_MODEL, client=None, now: Optional[float] = None,
          extra_files: Optional[Dict[str, str]] = None, dreams_dir: str = DREAMS, logs_dir: str = LOGS) -> Optional[str]:
    """Review everything since the last dream and write dreams/<date>.md. Returns its path."""
    now = now or time.time()
    state_path = os.path.join(dreams_dir, "state.json")
    state = load_state(state_path)
    since = float(state.get("covered_until") or now - 3 * 86400)
    digest = build_digest(since, face_dir, extra_files, logs_dir=logs_dir, dreams_dir=dreams_dir)
    if not digest["sessions"] and not digest["incidents"]:
        print("[dream] nothing new to review")
        state.update({"last_dream": time.strftime("%Y-%m-%d", time.localtime(now)), "covered_until": now})
        save_state(state, state_path)
        return None
    print(f"[dream] reviewing {digest['sessions']} sessions, {digest['incidents']} incidents "
          f"({digest['chars'] // 1000}k characters) with {model}")
    t0 = time.monotonic()
    result = review(digest["text"], model=model, client=client)
    took = time.monotonic() - t0
    day = time.strftime("%Y-%m-%d", time.localtime(now))
    os.makedirs(dreams_dir, exist_ok=True)
    path = os.path.join(dreams_dir, f"{day}.md")
    header = (f"# Dream {day}\n\n"
              f"Reviewed {digest['sessions']} sessions and {digest['incidents']} incidents since "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(since))} with {model} in {took / 60:.1f} min "
              f"({result['input_tokens']} tokens in, {result['output_tokens']} out).\n\n")
    if result["stop_reason"] == "refusal":
        body = "_The review was declined by the model's safety system; nothing to report._\n"
    else:
        body = result["text"] + ("\n\n_(Cut off at the output limit.)_\n" if result["stop_reason"] == "max_tokens" else "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + body)
    state.update({"last_dream": day, "covered_until": now, "last_report": os.path.basename(path),
                  "last_tokens": [result["input_tokens"], result["output_tokens"]]})
    save_state(state, state_path)
    print(f"[dream] wrote {path} ({took / 60:.1f} min)")
    return path


# ── when to dream ─────────────────────────────────────────────────────────────
class DreamScheduler:
    """Checks every `check_s`; dreams once per day when the character has been dormant and
    nobody has talked to it for `idle_hours`. The review runs on its own thread."""

    def __init__(self, run: Callable[[], Optional[str]], is_dormant: Callable[[], bool],
                 last_activity: Callable[[], float], idle_hours: float = 3.0, check_s: float = 600.0,
                 online: Callable[[], bool] = lambda: True, clock: Callable[[], float] = time.time,
                 state_path: str = STATE):
        self.run, self.is_dormant, self.last_activity = run, is_dormant, last_activity
        self.idle_hours, self.check_s, self.online, self.clock = idle_hours, check_s, online, clock
        self.state_path = state_path
        self.dreaming = False
        self._stop = threading.Event()

    def due(self) -> bool:
        now = self.clock()
        today = time.strftime("%Y-%m-%d", time.localtime(now))
        return (not self.dreaming and load_state(self.state_path).get("last_dream") != today
                and self.is_dormant() and self.online()
                and now - self.last_activity() >= self.idle_hours * 3600)

    def dream_now(self) -> str:
        if self.dreaming:
            return "already dreaming"
        threading.Thread(target=self._dream, name="dream", daemon=True).start()
        return "dreaming (the report appears under Dreams when done)"

    def _dream(self) -> None:
        self.dreaming = True
        try:
            self.run()
        except Exception as e:
            print(f"[dream] failed: {type(e).__name__}: {str(e)[:200]}")
        finally:
            self.dreaming = False

    def start(self) -> None:
        def loop():
            while not self._stop.wait(self.check_s):
                try:
                    if self.due():
                        self._dream()
                except Exception as e:
                    print(f"[dream] check failed: {e}")
        threading.Thread(target=loop, name="dream-scheduler", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def list_reports(dreams_dir: str = DREAMS) -> List[str]:
    return sorted((os.path.basename(p) for p in glob.glob(os.path.join(dreams_dir, "20*.md"))), reverse=True)


if __name__ == "__main__":
    import argparse
    from .env_config import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser(description="Review the logs now and write a dream report")
    ap.add_argument("--now", action="store_true", help="dream now, whatever the time")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--face-dir", default=os.path.join(HERE, "faces", "clara"))
    a = ap.parse_args()
    print(dream(face_dir=a.face_dir, model=a.model) or "no report")
