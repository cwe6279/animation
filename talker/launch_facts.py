"""
launch_facts.py — {placeholders} filled into a character's prompt when it starts.

A character.md (or the "character" string in face.json) may contain placeholders
that are replaced once, at launch:

    Today is {date}, the time is {time} {timezone}.

Built in: {date} {iso_date} {time} {datetime} {weekday} {year} {timezone} {utc_offset}.

Two ways to add your own. Per face, with no code, in face.json:

    "facts": {"venue": "the north gate", "event": "the Fall Festival"}

which gives {venue} and {event}. Or in code, for something that has to be looked
up at launch:

    from talker.launch_facts import register
    register("weather", lambda: fetch_forecast())

Unknown placeholders are left alone, and so are the {{move nod}} action blocks,
so a character file can use both.

Filled once, at launch, not per turn: the system prompt is cached by the brain
and rewriting it every turn would throw that cache away, costing latency on
every reply. A process left running past midnight keeps the date it started
with, so restart a long-lived kiosk daily, or give the character a clock tool.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Dict, Optional

# {name} but not {{name}}, so action blocks survive untouched.
_TOKEN = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")

_EXTRA: Dict[str, Callable[[], str]] = {}


def register(name: str, value: Callable[[], str]) -> None:
    """Add a placeholder of your own, resolved when a character loads."""
    _EXTRA[name.strip().lower()] = value


def builtin_facts(now: Optional[datetime] = None) -> Dict[str, str]:
    now = now or datetime.now().astimezone()
    offset = now.strftime("%z") or "+0000"
    # Month name before the day, with the ISO date beside it: day-first reads as another
    # date entirely to a model, and one did shift the day by one when given it that way.
    day = f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.strftime('%Y')}"
    return {
        "date": day,
        "iso_date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "datetime": f"{day} ({now.strftime('%Y-%m-%d')}) at {now.strftime('%H:%M')}",
        "weekday": now.strftime("%A"),
        "year": now.strftime("%Y"),
        "timezone": now.tzname() or "local time",
        "utc_offset": f"UTC{offset[:3]}:{offset[3:]}",
    }


def facts(extra: Optional[Dict[str, str]] = None, now: Optional[datetime] = None) -> Dict[str, str]:
    """Everything a placeholder can resolve to: built-ins, registered, then the face's own."""
    out = builtin_facts(now)
    for name, fn in _EXTRA.items():
        try:
            out[name] = str(fn())
        except Exception as e:
            print(f"[facts] {name} failed: {e}")
    for name, value in (extra or {}).items():
        out[str(name).strip().lower()] = str(value)
    return out


def expand(text: str, extra: Optional[Dict[str, str]] = None,
           now: Optional[datetime] = None) -> str:
    """Fill the placeholders in a character's prompt. Unknown ones are left as written."""
    if not text or "{" not in text:
        return text
    values = facts(extra, now)
    return _TOKEN.sub(lambda m: values.get(m.group(1), m.group(0)), text)
