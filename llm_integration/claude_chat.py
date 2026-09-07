"""
claude_chat.py — multi-turn Claude conversation that streams replies.

    chat = ClaudeChat(character="a curious little robot")
    for chunk in chat.reply("hello there"):   # yields text as it is generated
        ...

History is kept so follow-up questions work. The system prompt is the emotion
tag guide in system_prompt.md, so replies carry [emotion] tags the face can
use. Server-side refusal fallbacks are on so a declined request still gets
answered by another model.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))


def load_system_prompt() -> str:
    with open(os.path.join(HERE, "system_prompt.md"), encoding="utf-8") as f:
        text = f.read()
    # Everything below the "## System Prompt" heading *line* (the notes above it
    # mention the heading in passing, so match the line, not the phrase).
    import re
    m = re.search(r"^## System Prompt\s*$", text, re.M)
    return text[m.end():].strip() if m else text


VOICE_RULES = (
    "\n\nYou are talking out loud in a live conversation. Keep replies short, "
    "one to three sentences, and answer first. No lists, no markdown."
)

WAKE_RULES = (
    "\n\nVisitors start a conversation by saying your name. When a visitor says goodbye, thanks you "
    "and leaves, or the conversation is clearly finished, say a short farewell and end your reply "
    "with the exact marker [end] (it is not spoken). Otherwise never use that marker."
)

VISION_RULES = (
    "\n\nYou can see. Now and then an entry beginning 'You notice:' appears in the conversation. "
    "That is your own eyesight, an inner observation, not something anyone said and not text to "
    "read out. React to it the way a person reacts to what they see: a glance, a short remark in "
    "your own words, only if it matters right now. Never recite or paraphrase the observation "
    "itself, never say 'I notice' or 'I see that', and do not repeat details you have already "
    "mentioned. If you have no observation to go on, say you cannot make it out rather than invent."
)


def make_client() -> anthropic.Anthropic:
    """
    Anthropic client. Organization-wide API keys must name a workspace; set
    ANTHROPIC_WORKSPACE_ID (wrkspc_...) in .env, or use a key created inside
    a workspace in the Console and leave it unset.
    """
    headers = {}
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if ws:
        headers["anthropic-workspace-id"] = ws
    return anthropic.Anthropic(default_headers=headers or None)


class ClaudeChat:
    def __init__(self, model: str = "claude-opus-5", effort: str = "low",
                 character: Optional[str] = None, max_history: int = 20,
                 client: Optional[anthropic.Anthropic] = None, thinking: bool = True,
                 can_see: bool = False, wake_mode: bool = False):
        self.model = model
        self.effort = effort
        self.thinking = thinking     # False = no reasoning pass before answering (faster first token)
        self.max_history = max_history
        self.system = (load_system_prompt() + VOICE_RULES + (VISION_RULES if can_see else "")
                       + (WAKE_RULES if wake_mode else ""))
        if character:
            self.system += f"\n\nCharacter: {character}"
        self.messages: List[dict] = []
        self.last_usage = None
        # Scene context: pushed by the vision watcher only when the scene changed.
        # It is inserted into the conversation as its own context entry ahead of
        # the next thing the visitor says; quiet turns add nothing.
        self._pending_context: Optional[str] = None
        self.client = client or make_client()

    def add_context(self, text: str) -> None:
        """Queue a context note (e.g. a scene change). Only the latest one is kept."""
        self._pending_context = text.strip() or None

    def _push_user(self, user_text: str) -> None:
        if self._pending_context:
            self.messages.append({"role": "user",
                                  "content": f"(You notice: {self._pending_context})"})
            self._pending_context = None
        self.messages.append({"role": "user", "content": user_text})

    def reply(self, user_text: str) -> Iterator[str]:
        self._push_user(user_text)
        self.messages = self.messages[-self.max_history:]
        parts: List[str] = []
        try:
            extra = {}
            haiku = self.model.startswith("claude-haiku")
            if not haiku:   # Haiku 4.5 rejects effort, thinking-disabled and fallbacks
                extra["output_config"] = {"effort": self.effort}
                extra["betas"] = ["server-side-fallback-2026-07-01"]
                extra["fallbacks"] = "default"
                if not self.thinking:
                    extra["thinking"] = {"type": "disabled"}
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=512,
                **extra,
                # The system prompt is identical every turn: mark it cacheable so
                # the server reuses it (cheaper, and a little faster to first token).
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                messages=self.messages,
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
                    yield text
                self.last_usage = stream.get_final_message().usage
        finally:
            full = "".join(parts).strip()
            if full:
                self.messages.append({"role": "assistant", "content": full})
            else:
                self.messages.pop()      # failed turn: don't leave a dangling user message
