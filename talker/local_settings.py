"""
talker/local_settings.py — settings changed on the control page that should outlive the run.

    settings.json (repo root, gitignored, next to calibration.json)
    {"agent_url": "http://agentbox:8030"}

Not secrets: those stay in .env. A command-line flag still wins at startup; this
file only fills in what the flags leave unset, the same way calibration.json does.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(ROOT, "settings.json")


def load(path: str = SETTINGS_FILE) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(key: str, value: Any, path: str = SETTINGS_FILE) -> Dict[str, Any]:
    """Set one key (None removes it) and write the file atomically. Returns the new contents."""
    data = load(path)
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return data
