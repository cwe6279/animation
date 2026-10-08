"""
sleep_hours.py — the hours a character does not listen at all.

During sleep hours nothing goes to cloud speech recognition: the microphone is
not streamed and the connection is closed, so an empty room at night costs
nothing. News (a finished errand) waits until morning. "Wake now" on the
control page overrides it until the window ends.

    SleepHours("23:00-07:00").asleep(datetime.now())   # True from 23:00 to 06:59
    SleepHours("")                                      # never asleep
"""

from __future__ import annotations

import re
from datetime import datetime, time as dtime, timedelta
from typing import Optional, Tuple

DEFAULT = "23:00-07:00"
_SPEC = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def parse(spec: str) -> Optional[Tuple[dtime, dtime]]:
    """'23:00-07:00' -> (start, end); '' or 'off' -> None. Raises ValueError on nonsense."""
    if not spec or spec.strip().lower() in ("off", "none", "never"):
        return None
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"sleep hours look like 23:00-07:00, not {spec!r}")
    h1, m1, h2, m2 = map(int, m.groups())
    if not (0 <= h1 < 24 and 0 <= h2 < 24 and m1 < 60 and m2 < 60):
        raise ValueError(f"not a time of day: {spec!r}")
    return dtime(h1, m1), dtime(h2, m2)


class SleepHours:
    def __init__(self, spec: str = DEFAULT):
        self.spec = ""
        self.window: Optional[Tuple[dtime, dtime]] = None
        self.awake_until: Optional[datetime] = None     # "wake now" until this moment
        self.set(spec)

    def set(self, spec: str) -> None:
        self.window = parse(spec)
        self.spec = spec.strip() if self.window else ""
        self.awake_until = None

    def in_window(self, now: datetime) -> bool:
        if self.window is None:
            return False
        start, end, t = self.window[0], self.window[1], now.time()
        if start == end:
            return False
        return start <= t < end if start < end else (t >= start or t < end)

    def asleep(self, now: datetime) -> bool:
        if self.awake_until is not None:
            if now < self.awake_until:
                return False
            self.awake_until = None
        return self.in_window(now)

    def ends(self, now: datetime) -> Optional[datetime]:
        """When the current (or next) sleep window ends."""
        if self.window is None:
            return None
        end = datetime.combine(now.date(), self.window[1], tzinfo=now.tzinfo)
        return end if end > now else end + timedelta(days=1)

    def wake_now(self, now: datetime) -> None:
        """Stay awake until the current window ends; the next night sleeps as usual."""
        self.awake_until = self.ends(now) if self.in_window(now) else None
