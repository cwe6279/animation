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
import time
from typing import Iterator, List, Optional

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))

# Fast mode (research preview): the same model at up to 2.5x the output tokens per
# second, at double the price. Add "-fast" to the model name: claude-opus-5-fast.
# It speeds up tokens per second, NOT time to first token, so in this pipeline it
# shortens the wait for the first *sentence* to reach the voice, not the first token.
# Opus 5 and Opus 4.8 only; 4.7 errors and 4.6 quietly runs standard.
FAST_BETA = "fast-mode-2026-02-01"
FAST_MODELS = ("claude-opus-5", "claude-opus-4-8")


def split_speed(model: str) -> tuple:
    """('claude-opus-5-fast') -> ('claude-opus-5', 'fast'). Unknown models keep their name."""
    if not model or not model.endswith("-fast"):
        return model, None
    base = model[: -len("-fast")]
    if base not in FAST_MODELS:
        print(f"[brain] fast mode is only offered on {' and '.join(FAST_MODELS)}; "
              f"running {base} at standard speed")
        return base, None
    return base, "fast"


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


def assistant_rules(memory: bool = False, errands: bool = False, can: str = "") -> str:
    """The system-prompt addition for an assistant with a working memory (notes.md,
    tasks.md) and/or a backend agent to hand work to. '' when it has neither."""
    if not (memory or errands):
        return ""
    lines = [
        "\n\nA line in parentheses that begins 'You notice:' or 'Event:' comes from the system, never "
        "from the person you are talking to: an observation, a tool's answer, or news that something "
        "finished. Never read it out or say 'I notice'; act on it in your own words. Every reply "
        "must contain words to say out loud: a {{...}} block on its own is silence to the listener, "
        "so put a short spoken sentence beside it ('Let me check.' / 'On it.')."
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
        if can:
            lines.append(
                f"Through that agent you can: {can}. You cannot do any of those yourself in the room, "
                "so when someone asks for one of them, hand it off with {{task ...}} and say you have. "
                "Never say you cannot, and never tell the person to look it up, check an app or do it "
                "themselves: getting the information is your job, through the agent."
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
    FAST_PAUSE_S = 300.0             # after three fast-mode rate limits in a row, wait this long

    def __init__(self, model: str = "claude-opus-5", effort: str = "low",
                 character: Optional[str] = None, max_history: int = 20,
                 client: Optional[anthropic.Anthropic] = None, thinking: bool = True,
                 can_see: bool = False, wake_mode: bool = False, extra_rules: str = ""):
        self.model, self.speed = split_speed(model)
        self.effort = effort
        self.thinking = thinking     # False = no reasoning pass before answering (faster first token)
        self.max_history = max_history
        self.system = (load_system_prompt() + VOICE_RULES + (VISION_RULES if can_see else "")
                       + (WAKE_RULES if wake_mode else "") + extra_rules)
        if character:
            self.system += f"\n\nCharacter: {character}"
        self.messages: List[dict] = []
        self.last_usage = None
        self._fast_429 = 0           # consecutive fast-mode rate limits
        self._fast_pause_until = 0.0  # after a run of them, stop asking for a while
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

    def _stream(self, speed: Optional[str]):
        extra = {}
        haiku = self.model.startswith("claude-haiku")
        if not haiku:   # Haiku 4.5 rejects effort, thinking-disabled and fallbacks
            extra["output_config"] = {"effort": self.effort}
            extra["betas"] = ["server-side-fallback-2026-07-01"]
            extra["fallbacks"] = "default"
            if not self.thinking:
                extra["thinking"] = {"type": "disabled"}
        client = self.client
        if speed == "fast":
            extra["speed"] = "fast"
            extra["betas"] = list(extra.get("betas", [])) + [FAST_BETA]
            # Don't sit through the SDK's retry backoff when fast capacity is gone:
            # fail at once and let the caller drop to standard speed for this turn.
            opts = getattr(client, "with_options", None)
            client = opts(max_retries=0) if opts else client
        return client.beta.messages.stream(
            model=self.model,
            max_tokens=512,
            **extra,
            # The system prompt is identical every turn: mark it cacheable so
            # the server reuses it (cheaper, and a little faster to first token).
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
        )

    def reply(self, user_text: str) -> Iterator[str]:
        self._push_user(user_text)
        self.messages = self.messages[-self.max_history:]
        parts: List[str] = []
        speed = self.speed
        if speed == "fast" and time.monotonic() < self._fast_pause_until:
            speed = None                 # backing off after a run of rate limits
        try:
            while True:
                try:
                    with self._stream(speed) as stream:
                        for text in stream.text_stream:
                            parts.append(text)
                            yield text
                        self.last_usage = stream.get_final_message().usage
                    if speed == "fast":
                        self._fast_429 = 0
                    break
                except Exception as e:
                    # Nothing has been said yet and fast mode is what failed: say it
                    # at standard speed rather than not at all. A rate limit is this
                    # turn only (capacity returns in seconds); anything else — no
                    # access to the preview, a bad beta — is for the whole session.
                    if speed != "fast" or parts:
                        raise
                    import anthropic as _a
                    if isinstance(e, _a.RateLimitError):
                        # Fast capacity replenishes continuously, so a 429 is usually
                        # over in seconds: answer this turn at standard speed and try
                        # again next turn. Only a run of them is worth backing off from.
                        self._fast_429 += 1
                        # "rate limit of 0 fast mode input tokens" means this key has no
                        # fast allocation (it is a research preview, granted per account),
                        # so don't spend two more turns discovering that.
                        if "of 0 fast mode" in str(e):
                            self._fast_429 = 3
                        if self._fast_429 >= 3:
                            self._fast_pause_until = time.monotonic() + self.FAST_PAUSE_S
                            print(f"[brain] no fast mode capacity for a request this size; standard "
                                  f"speed for {self.FAST_PAUSE_S / 60:.0f} minutes. Fast mode is a "
                                  f"research preview your key has to be granted")
                        else:
                            print("[brain] fast mode is rate limited; this reply goes at standard speed")
                    else:
                        print(f"[brain] fast mode unavailable ({e}); standard speed from here on")
                        self.speed = None
                    speed = None
        finally:
            full = "".join(parts).strip()
            if full:
                self.messages.append({"role": "assistant", "content": full})
            else:
                self._drop_failed_turn()
