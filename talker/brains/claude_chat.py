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


def assistant_rules(memory: bool = False, errands: bool = False) -> str:
    """The system-prompt addition for an assistant with a working memory (notes.md,
    tasks.md) and/or a backend agent to hand work to. '' when it has neither."""
    if not (memory or errands):
        return ""
    lines = [
        "\n\nA line in parentheses that begins 'You notice:' or 'Event:' comes from the system, never "
        "from the person you are talking to: an observation, a tool's answer, or news that something "
        "finished. Never read it out or say 'I notice'; act on it in your own words."
    ]
    if memory:
        lines.append(
            "You have a working memory in two files you read at launch: your notes and your task ledger, "
            "both in your brief below. Write a note with {{note <one line>}} anywhere in a reply; it is "
            "appended to your notes with the time, and you will read it again next time you start. Note "
            "what is worth keeping across sessions: decisions, names, preferences, follow-ups, where you "
            "left off. Tell the person in a few words that you noted it. When a session ends you are "
            "asked for a short summary; keep it factual."
        )
    if errands:
        lines.append(
            "You can hand work to a separate agent on another machine that researches, reads, writes and "
            "runs code for minutes at a time: {{task <what to do, one clear sentence with everything it "
            "needs>}}. Use it only for work that genuinely takes longer than a reply, never for something "
            "you can answer now. The block is silent, so the same reply must also say, in words, that you "
            "have handed it off, like 'On it. I'll tell you when it's back.' When it finishes you get an "
            "Event with a short summary: tell the person the conclusion in your own words. The ledger in "
            "your brief shows each task's state; {{tool tasks}} reads it live when someone asks what is "
            "open, and like every block it is never spoken and never offered as something to do."
        )
    return "\n".join(lines)


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
                 can_see: bool = False, wake_mode: bool = False, extra_rules: str = ""):
        self.model = model
        self.effort = effort
        self.thinking = thinking     # False = no reasoning pass before answering (faster first token)
        self.max_history = max_history
        self.system = (load_system_prompt() + VOICE_RULES + (VISION_RULES if can_see else "")
                       + (WAKE_RULES if wake_mode else "") + extra_rules)
        if character:
            self.system += f"\n\nCharacter: {character}"
        self.messages: List[dict] = []
        self.last_usage = None
        # Context notes (a scene change, a tool's answer, a finished task) wait here
        # and go into the conversation as one entry ahead of the next thing the
        # visitor says; quiet turns add nothing. All of them are kept, in order.
        self._pending_context: List[str] = []
        self.client = client or make_client()

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
        """A turn that produced nothing: remove its user line and any notice pushed with it."""
        self.messages.pop()
        if self.messages and self.messages[-1]["role"] == "user" \
                and self.messages[-1]["content"].startswith("(You notice: "):
            self.messages.pop()

    def summarise(self, instruction: str, max_tokens: int = 300) -> str:
        """One extra answer about the conversation so far (for the session note).
        Leaves the history untouched."""
        msgs = self.messages + [{"role": "user", "content": instruction}]
        if not self.messages:
            return ""
        extra = {}
        if not self.model.startswith("claude-haiku"):
            extra["thinking"] = {"type": "disabled"}
        with self.client.beta.messages.stream(model=self.model, max_tokens=max_tokens, **extra,
                                              system=self.system, messages=msgs) as stream:
            return "".join(stream.text_stream).strip()

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
                self._drop_failed_turn()
