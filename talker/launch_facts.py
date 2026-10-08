"""
launch_facts.py — {placeholders} filled into a character's prompt when it starts.

A character.md (or the "character" string in face.json) may contain placeholders
that are replaced once, at launch:

    Today is {date}, the time is {time} {timezone}.

Built in: {date} {iso_date} {time} {datetime} {weekday} {year} {timezone} {utc_offset},
and {location}: the "location" saved in settings.json (the control page sets it), else the
city looked up once from this machine's internet address, else "an unknown location".

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


_located: Dict[str, str] = {}


def location(lookup: bool = True) -> str:
    """Where the character is: settings.json "location", else an IP-based city (looked up once
    per run, 3 s timeout), else 'an unknown location'. Never raises."""
    try:
        from .local_settings import load
        saved = str(load().get("location") or "").strip()
        if saved:
            return saved
    except Exception:
        pass
    if "ip" not in _located and lookup:
        _located["ip"] = ""
        try:
            import json
            import urllib.request
            with urllib.request.urlopen("https://ipinfo.io/json", timeout=3) as r:
                d = json.load(r)
            _located["ip"] = ", ".join(p for p in (d.get("city"), d.get("region"), d.get("country")) if p)
        except Exception:
            pass
    return _located.get("ip") or "an unknown location"


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
    if "{location}" in text and "location" not in values:
        values["location"] = location()           # only looked up when a prompt asks for it
    return _TOKEN.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def now_text(now: Optional[datetime] = None) -> str:
    """The current date, time, zone and place, as one line for the brain."""
    f = builtin_facts(now)
    return f"It is now {f['datetime']} {f['timezone']} ({f['utc_offset']}), in {location()}."
