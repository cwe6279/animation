"""
talker/frame_governor.py — adaptive frame rate for weak hardware (a Raspberry Pi).

Measures how long each frame's work takes (update + draw + flip). If frames
keep running over the budget for the current target rate, the target steps
down (60 -> 45 -> 30 -> 20 -> 15). Once frames have had clear headroom for a
while, it steps back up. Lip-sync timing comes from the audio clock, so a
lower frame rate only changes smoothness, never sync.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

STEPS = [60, 45, 30, 20, 15]


class FrameGovernor:
    def __init__(self, target_fps: int = 60, min_fps: int = 15, enabled: bool = True,
                 window: int = 30, over_ratio: float = 0.9, under_ratio: float = 0.5,
                 recover_after_s: float = 5.0, on_change: Optional[Callable[[int, int, float], None]] = None):
        self.enabled = enabled
        self.max_fps = target_fps
        self.min_fps = min_fps
        self.fps = target_fps
        self._steps = [s for s in STEPS if min_fps <= s <= target_fps] or [target_fps]
        if target_fps not in self._steps:
            self._steps.insert(0, target_fps)
        self._times: deque = deque(maxlen=window)
        self._over = over_ratio          # frame time above this fraction of the budget = too slow
        self._under = under_ratio        # below this fraction of the *next higher* budget = room to go up
        self._recover_after = recover_after_s
        self._calm_seconds = 0.0
        self.on_change = on_change or (lambda old, new, avg: print(
            f"[fps] {old} -> {new} (frames averaging {avg*1000:.1f} ms)"))
        self.last_avg = 0.0

    def record(self, work_seconds: float, dt: float) -> int:
        """Call once per frame with the time the frame's work took. Returns the target fps."""
        if not self.enabled:
            return self.fps
        self._times.append(work_seconds)
        if len(self._times) < self._times.maxlen:
            return self.fps
        avg = sum(self._times) / len(self._times)
        self.last_avg = avg
        budget = 1.0 / self.fps
        idx = self._steps.index(self.fps)
        if avg > budget * self._over and idx < len(self._steps) - 1:
            old, self.fps = self.fps, self._steps[idx + 1]
            self._times.clear()
            self._calm_seconds = 0.0
            self.on_change(old, self.fps, avg)
        elif idx > 0:
            higher = self._steps[idx - 1]
            if avg < (1.0 / higher) * self._under:
                self._calm_seconds += dt
                if self._calm_seconds >= self._recover_after:
                    old, self.fps = self.fps, higher
                    self._times.clear()
                    self._calm_seconds = 0.0
                    self.on_change(old, self.fps, avg)
            else:
                self._calm_seconds = 0.0
        return self.fps
