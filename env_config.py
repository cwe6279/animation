"""
env_config.py — load secrets from a .env file next to the code.

    ELEVENLABS_API_KEY=...      (aliases accepted: ELEVEN_LABS_API, ELEVEN_LABS_API_KEY, ELEVEN_API_KEY)
    ANTHROPIC_API_KEY=...
    ANTHROPIC_WORKSPACE_ID=wrkspc_...   (only for organization-wide keys)

Values already present in the environment win. The file is gitignored.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

ALIASES = {
    "ELEVENLABS_API_KEY": ["ELEVEN_LABS_API", "ELEVEN_LABS_API_KEY", "ELEVEN_API_KEY", "ELEVENLABS_KEY"],
    "ANTHROPIC_API_KEY": ["ANTHROPIC_KEY", "CLAUDE_API_KEY"],
    "GROQ_API_KEY": ["GROQ_KEY", "GROQ_API"],
    "OPENAI_API_KEY": ["OPENAI_KEY", "OPENAI_API"],
}


def load_dotenv(path: str | None = None) -> dict:
    """Parse KEY=value lines (quotes and `export` allowed) into os.environ. Returns what was set."""
    path = path or os.path.join(HERE, ".env")
    loaded: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded[key] = value
    for canonical, names in ALIASES.items():
        if not os.environ.get(canonical):
            for alias in names:
                if os.environ.get(alias):
                    os.environ[canonical] = os.environ[alias]
                    break
    return loaded
