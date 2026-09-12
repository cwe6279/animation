"""
openai_compat_chat.py — streaming multi-turn chat over the OpenAI API.

Same interface as ClaudeChat: reply(text) yields chunks, history is kept.
Used for side-by-side comparisons; Claude is the default brain.

    chat = OpenAICompatChat.openai(model="gpt-4o-mini", character="...")

Uses the same system prompt (emotion/performance tags) as ClaudeChat.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

from openai import OpenAI

from .claude_chat import VISION_RULES, VOICE_RULES, WAKE_RULES, load_system_prompt

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatChat:
    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str] = None,
                 character: Optional[str] = None, max_history: int = 20, client=None,
                 temperature: float = 0.8, name: str = "openai", can_see: bool = False,
                 wake_mode: bool = False, extra_rules: str = ""):
        self.name = name
        self.model = model
        self.max_history = max_history
        self.temperature = temperature
        self.system = (load_system_prompt() + VOICE_RULES + (VISION_RULES if can_see else "")
                       + (WAKE_RULES if wake_mode else "") + extra_rules)
        if character:
            self.system += f"\n\nCharacter: {character}"
        self.messages: List[dict] = []
        self.last_usage = None
        # Context notes wait here and go in as one entry ahead of the next thing
        # the visitor says; quiet turns add nothing. All of them are kept, in order.
        self._pending_context: List[str] = []
        if client is None:
            if not api_key:
                raise RuntimeError(f"{name} needs an API key in the environment")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client

    @classmethod
    def openai(cls, model: Optional[str] = None, character: Optional[str] = None, **kw):
        return cls(model or OPENAI_DEFAULT_MODEL, os.environ.get("OPENAI_API_KEY"), None,
                   character=character, name="openai", **kw)

    def add_context(self, text: str) -> None:
        """Queue a context note for the next turn."""
        text = text.strip()
        if text and text not in self._pending_context:
            self._pending_context.append(text)

    def _push_user(self, user_text: str) -> None:
        if self._pending_context:
            self.messages.append({"role": "user",
                                  "content": "(You notice: " + "\n".join(self._pending_context) + ")"})
            self._pending_context = []
        self.messages.append({"role": "user", "content": user_text})

    def _drop_failed_turn(self) -> None:
        self.messages.pop()
        if self.messages and self.messages[-1]["role"] == "user" \
                and self.messages[-1]["content"].startswith("(You notice: "):
            self.messages.pop()

    def summarise(self, instruction: str, max_tokens: int = 300) -> str:
        """One extra answer about the conversation so far; history is left untouched."""
        if not self.messages:
            return ""
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0.3,
            messages=[{"role": "system", "content": self.system}] + self.messages
                     + [{"role": "user", "content": instruction}])
        return (r.choices[0].message.content or "").strip()

    def reply(self, user_text: str) -> Iterator[str]:
        self._push_user(user_text)
        self.messages = self.messages[-self.max_history:]
        parts: List[str] = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system}] + self.messages,
                max_tokens=256,
                temperature=self.temperature,
                stream=True,
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
                self._drop_failed_turn()
