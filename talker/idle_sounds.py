"""
idle_sounds.py — ambient sounds while nothing is happening.

Files in faces/<name>/sounds/idle/ (wav, ogg, mp3) are played one at a time, at
random intervals, whenever the character has been quiet for a while: not
speaking, not thinking, nobody talking. A purr, a creak, a distant cackle. They
are separate from the sounds/ folder the brain can call with {{sfx name}}, so
ambience never shows up in the prompt.

face.json:  "idle_sounds": {"interval": [30, 90], "quiet_for": 10}
            seconds between sounds (random in the range) and how long the room
            must have been quiet first. Both optional; the folder is the switch.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, Sequence

from .actions import SoundBank


class IdleSounds:
    def __init__(self, bank: SoundBank, is_quiet: Callable[[], bool],
                 interval: Sequence[float] = (30.0, 90.0), quiet_for: float = 10.0,
                 clock: Callable[[], float] = time.monotonic, rng: Optional[random.Random] = None):
        self.bank = bank
        self.is_quiet = is_quiet
        self.interval = (float(interval[0]), float(interval[1]))
        self.quiet_for = float(quiet_for)
        self.clock = clock
        self.rng = rng or random.Random()
        self.enabled = bool(bank.names)
        self.played = 0
        self.last_name: Optional[str] = None
        self._quiet_since: Optional[float] = None
        self._next_at = self.clock() + self._gap()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _gap(self) -> float:
        return self.rng.uniform(*self.interval)

    def tick(self) -> Optional[str]:
        """Call regularly. Plays a sound when due and the room has been quiet long
        enough; returns its name, else None."""
        if not self.enabled:
            return None
        now = self.clock()
        if not self.is_quiet():
            self._quiet_since = None
            return None
        if self._quiet_since is None:
            self._quiet_since = now
        if now < self._next_at or now - self._quiet_since < self.quiet_for:
            return None
        names = [n for n in self.bank.names if n != self.last_name] or self.bank.names
        name = self.rng.choice(names)
        self._next_at = now + self._gap()
        if self.bank.play(name):
            self.played += 1
            self.last_name = name
            return name
        return None

    # a small thread so voice_loop needs no timer of its own
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="idle-sounds", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.tick()
            except Exception as e:                    # ambience must never break the loop
                print(f"[idle] {e}")

    def stop(self) -> None:
        self._stop.set()
