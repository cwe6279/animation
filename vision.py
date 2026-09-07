"""
vision.py — a camera watcher that takes notes about the scene for the brain.

    python voice_loop.py --list-cameras
    python voice_loop.py --face eve --camera c920            # or an index: --camera 0

Every `interval` seconds (default 9) the watcher grabs `burst` frames
(default 3, ~0.3 s apart), downsizes them, and sends them to a fast vision
model which returns a few lines of notes: who is there, what they are doing,
anything notable. Only the latest notes are kept; frames are discarded as
soon as they have been described. If the model flags an emergency (someone
hurt, a fire, a child alone in distress) the burst and the note are written
to `emergencies/<timestamp>/` and the notes are marked so the brain can react.

The notes are handed to the brain on the next turn as "what you can see",
so the character can greet people, count them, and refer to what they hold.

Privacy: nothing is stored except emergencies. Notes are text only.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

VISION_PROMPT = """You are the eyes of an animated character that talks with visitors (it may be a kids' event, a museum, a library). You get {n} frames taken about 0.3 s apart from its camera. Write short notes the character can use in conversation.

Reply with JSON only:
{{"notes": "<2-4 short sentences: how many people, rough ages (child/teen/adult), what they are doing, ANY object held up or shown to the camera (name it, its colour, any text on it), costumes or headwear, mood; say 'nobody in view' if empty>",
 "people": <integer>,
 "emergency": <true|false>,
 "emergency_reason": "<only if emergency: what you see>"}}

Emergency means someone appears hurt, in danger, or in real distress, or there is fire/smoke or a clear hazard. Never guess identities. Keep notes factual and kind."""


@dataclass
class SceneNote:
    time: float                      # time.monotonic() when the burst was taken
    notes: str
    people: int = 0
    emergency: bool = False
    emergency_reason: str = ""
    wall_time: str = ""

    def age_s(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else now) - self.time


# ─────────────────────────────────────────────────────
# CAMERAS
# ─────────────────────────────────────────────────────
def list_cameras() -> List[dict]:
    """[{index, name, path}] for capture devices. Linux reads names from sysfs."""
    cams = []
    if os.path.isdir("/sys/class/video4linux"):
        for d in sorted(glob.glob("/sys/class/video4linux/video*")):
            idx = int(re.sub(r"\D", "", os.path.basename(d)) or 0)
            try:
                name = open(os.path.join(d, "name")).read().strip()
            except OSError:
                name = f"video{idx}"
            # a device exposes several nodes; only those with a capture capability are cameras
            caps = ""
            try:
                import subprocess
                caps = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{idx}", "--all"],
                                      capture_output=True, text=True, timeout=3).stdout
            except Exception:
                pass
            if caps and "Video Capture" not in caps.split("Device Caps")[-1][:400]:
                continue
            cams.append({"index": idx, "name": name, "path": f"/dev/video{idx}"})
    else:
        import cv2
        for idx in range(8):
            cap = cv2.VideoCapture(idx)
            ok = cap.isOpened()
            cap.release()
            if ok:
                cams.append({"index": idx, "name": f"camera {idx}", "path": str(idx)})
    return cams


def resolve_camera(spec) -> int:
    """Index, digit string, or a case-insensitive name fragment."""
    if spec is None or spec == "":
        cams = list_cameras()
        if not cams:
            raise RuntimeError("no camera found")
        return cams[0]["index"]
    if isinstance(spec, int) or str(spec).strip().isdigit():
        return int(spec)
    needle = str(spec).lower()
    for c in list_cameras():
        if needle in c["name"].lower():
            return c["index"]
    names = ", ".join(c["name"] for c in list_cameras())
    raise RuntimeError(f"no camera matching {spec!r}; available: {names}")


# ─────────────────────────────────────────────────────
# FRAME SOURCE
# ─────────────────────────────────────────────────────
class CameraSource:
    """Keeps the camera open and returns JPEG-encoded, downsized bursts."""

    def __init__(self, index: int, width: int = 896, height: int = 504, jpeg_quality: int = 80):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.size = (width, height)
        self.quality = jpeg_quality
        for _ in range(5):                       # let exposure settle
            self.cap.read()

    def burst(self, n: int = 3, spacing_s: float = 0.3) -> List[bytes]:
        frames: List[bytes] = []
        for i in range(n):
            for _ in range(3):                   # drain stale buffered frames
                self.cap.grab()
            ok, frame = self.cap.read()
            if ok:
                small = self.cv2.resize(frame, self.size, interpolation=self.cv2.INTER_AREA)
                ok2, buf = self.cv2.imencode(".jpg", small, [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if ok2:
                    frames.append(buf.tobytes())
            if i < n - 1:
                time.sleep(spacing_s)
        return frames

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


# ─────────────────────────────────────────────────────
# CHANGE DETECTION  (don't pay for a description of an unchanged scene)
# ─────────────────────────────────────────────────────
def frame_signature(jpeg: bytes, size=(24, 14)):
    """Tiny grayscale thumbnail of a JPEG, as floats 0..1, for cheap comparison."""
    import numpy as np
    try:
        import cv2
        arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None
        small = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0
    except Exception:
        return None


def change_score(a, b) -> float:
    """Mean absolute difference between two signatures (0 = identical, 1 = opposite)."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    import numpy as np
    return float(np.mean(np.abs(a - b)))


# ─────────────────────────────────────────────────────
# VISION MODEL
# ─────────────────────────────────────────────────────
def describe_with_claude(frames: List[bytes], model: str = "claude-haiku-4-5", client=None) -> dict:
    """Send JPEG frames to Claude and return the parsed JSON note."""
    import anthropic
    if client is None:
        from llm_integration.claude_chat import make_client
        client = make_client()
    content = []
    for jpg in frames:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(jpg).decode()}})
    content.append({"type": "text", "text": VISION_PROMPT.format(n=len(frames))})
    resp = client.messages.create(model=model, max_tokens=300,
                                  messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0) if m else text)
    except Exception:
        data = {"notes": text.strip()[:400], "people": 0, "emergency": False}
    return data


# ─────────────────────────────────────────────────────
# WATCHER
# ─────────────────────────────────────────────────────
class SceneWatcher:
    """
    Background loop: every `interval` s take a burst, describe it, keep the
    latest note. `describe(frames) -> dict` and `source.burst(n) -> frames`
    are injectable for tests.
    """

    def __init__(self, source, describe: Callable[[List[bytes]], dict], interval: float = 9.0,
                 burst: int = 3, keep: int = 3, emergency_dir: Optional[str] = None,
                 on_note: Optional[Callable[[SceneNote], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None, clock=time.monotonic,
                 change_threshold: float = 0.035, max_quiet_s: float = 90.0,
                 signature: Callable = frame_signature):
        self.source = source
        self.describe = describe
        self.interval = interval
        self.burst = burst
        self.keep = keep
        # Skip the model when the scene has not changed: compare a tiny thumbnail
        # of the new burst with the one last described. 0.06 ~ someone entering,
        # sitting down or waving; camera noise and lighting flicker stay below it.
        self.change_threshold = change_threshold
        self.max_quiet_s = max_quiet_s          # describe anyway after this long
        self.signature = signature
        self._last_sig = None
        self._last_described_at: Optional[float] = None
        self.emergency_dir = emergency_dir or os.path.join(HERE, "emergencies")
        self.on_note = on_note or (lambda n: None)
        self.on_error = on_error or (lambda msg: print(f"[vision] {msg}"))
        self.clock = clock
        self._notes: List[SceneNote] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()          # set by request() to observe now
        self._force = False
        self._done = threading.Condition()
        self._observations = 0                  # completed observe_once() calls
        self._thread: Optional[threading.Thread] = None
        self.stats = {"bursts": 0, "described": 0, "skipped_unchanged": 0, "errors": 0,
                      "emergencies": 0, "last_ms": 0, "last_change": 0.0}

    # ── lifecycle ──
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="scene-watcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        try:
            self.source.close()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            force, self._force = self._force, False
            self._wake.clear()
            try:
                self.observe_once(force=force)
            except Exception as e:
                self.stats["errors"] += 1
                self.on_error(f"observation failed: {e}")
            with self._done:
                self._observations += 1
                self._done.notify_all()
            elapsed = time.monotonic() - t0
            self._wake.wait(max(0.5, self.interval - elapsed)) if not self._stop.is_set() else None

    # ── on-demand ──
    def request(self, force: bool = True) -> int:
        """
        Ask for an observation now (e.g. the visitor just started talking).
        Returns a ticket for wait_for(); force=True bypasses change detection.
        """
        with self._done:
            ticket = self._observations + 1
        self._force = self._force or force
        self._wake.set()
        return ticket

    def wait_for(self, ticket: int, timeout: float) -> bool:
        """Block until the observation `ticket` (from request) has completed."""
        deadline = time.monotonic() + timeout
        with self._done:
            while self._observations < ticket:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._done.wait(remaining)
        return True

    # ── one observation ──
    def observe_once(self, force: bool = False) -> Optional[SceneNote]:
        frames = self.source.burst(self.burst)
        if not frames:
            return None
        self.stats["bursts"] += 1
        now = self.clock()
        sig = self.signature(frames[-1]) if self.signature else None
        score = change_score(sig, self._last_sig) if self._last_sig is not None else 1.0
        self.stats["last_change"] = round(score, 3)
        quiet_for = (now - self._last_described_at) if self._last_described_at is not None else None
        unchanged = (not force and score < self.change_threshold and quiet_for is not None
                     and quiet_for < self.max_quiet_s and self.latest() is not None)
        if unchanged:
            # Same scene: keep the existing note current instead of paying for a new one.
            self.stats["skipped_unchanged"] += 1
            with self._lock:
                self._notes[-1].time = now
            return None
        t0 = time.monotonic()
        data = self.describe(frames)
        self.stats["last_ms"] = round((time.monotonic() - t0) * 1000)
        self.stats["described"] += 1
        self._last_sig = sig
        self._last_described_at = now
        note = SceneNote(time=self.clock(), notes=str(data.get("notes", "")).strip(),
                         people=int(data.get("people") or 0), emergency=bool(data.get("emergency")),
                         emergency_reason=str(data.get("emergency_reason") or ""),
                         wall_time=time.strftime("%Y-%m-%d %H:%M:%S"))
        if note.emergency:
            self._save_emergency(frames, note)
        with self._lock:
            self._notes.append(note)
            del self._notes[:-self.keep]
        self.on_note(note)
        return note                      # frames go out of scope here: nothing kept

    def _save_emergency(self, frames: List[bytes], note: SceneNote) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = os.path.join(self.emergency_dir, stamp)
        os.makedirs(folder, exist_ok=True)
        for i, jpg in enumerate(frames):
            with open(os.path.join(folder, f"frame_{i + 1}.jpg"), "wb") as f:
                f.write(jpg)
        with open(os.path.join(folder, "note.json"), "w", encoding="utf-8") as f:
            json.dump({"time": note.wall_time, "notes": note.notes, "people": note.people,
                       "emergency_reason": note.emergency_reason}, f, indent=2)
        self.stats["emergencies"] += 1
        self.on_error(f"EMERGENCY flagged, frames saved to {folder}: {note.emergency_reason}")

    # ── what the brain gets ──
    def latest(self) -> Optional[SceneNote]:
        with self._lock:
            return self._notes[-1] if self._notes else None

    def context(self, max_age_s: float = 40.0) -> str:
        """Text for the brain: the latest note if it is fresh, plus an emergency flag."""
        note = self.latest()
        if note is None:
            return ""
        age = note.age_s(self.clock())
        if age > max_age_s:
            return ""
        s = f"What you can see right now (camera notes, {age:.0f}s old): {note.notes}"
        if note.emergency:
            s += f"\nEMERGENCY in view: {note.emergency_reason}. Stay calm, tell an adult to help, keep it short."
        return s
