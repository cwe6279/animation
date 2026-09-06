"""
openai_compat_chat.py — streaming multi-turn chat over any OpenAI-compatible API.

Covers OpenAI itself and Groq (Llama etc. on Groq's LPUs), which is worth a
look for a voice loop because of its time to first token. Same interface as
ClaudeChat: reply(text) yields chunks, history is kept.

    chat = OpenAICompatChat.groq(model="llama-3.3-70b-versatile", character="a spooky cat")
    chat = OpenAICompatChat.openai(model="gpt-4o-mini", character="...")

Uses the same system prompt (emotion/performance tags) as ClaudeChat.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

from openai import OpenAI

try:
    from .claude_chat import VOICE_RULES, load_system_prompt
except ImportError:
    from claude_chat import VOICE_RULES, load_system_prompt

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"     # ~150 ms to first token, in character
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatChat:
    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str] = None,
                 character: Optional[str] = None, max_history: int = 20, client=None,
                 temperature: float = 0.8, name: str = "openai"):
        self.name = name
        self.model = model
        self.max_history = max_history
        self.temperature = temperature
        self.system = load_system_prompt() + VOICE_RULES
        if character:
            self.system += f"\n\nCharacter: {character}"
        self.messages: List[dict] = []
        self.last_usage = None
        if client is None:
            if not api_key:
                raise RuntimeError(f"{name} needs an API key in the environment")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client

    @classmethod
    def groq(cls, model: Optional[str] = None, character: Optional[str] = None, **kw):
        return cls(model or GROQ_DEFAULT_MODEL, os.environ.get("GROQ_API_KEY"), GROQ_BASE_URL,
                   character=character, name="groq", **kw)

    @classmethod
    def openai(cls, model: Optional[str] = None, character: Optional[str] = None, **kw):
        return cls(model or OPENAI_DEFAULT_MODEL, os.environ.get("OPENAI_API_KEY"), None,
                   character=character, name="openai", **kw)

    def reply(self, user_text: str) -> Iterator[str]:
        self.messages.append({"role": "user", "content": user_text})
        self.messages = self.messages[-self.max_history:]
        parts: List[str] = []
        try:
            extra = {}
            if "gpt-oss" in self.model:
                extra["reasoning_effort"] = "low"    # otherwise it spends the budget thinking
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system}] + self.messages,
                max_tokens=256,
                temperature=self.temperature,
                stream=True,
                **extra,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield delta
        finally:
            full = "".join(parts).strip()
            if full:
                self.messages.append({"role": "assistant", "content": full})
            else:
                self.messages.pop()
