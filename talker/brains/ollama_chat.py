"""
ollama_chat.py — a local brain through Ollama (https://ollama.com), no key, no cloud.

    chat = OllamaChat(model="<name from --list-models>", character="a curious robot")
    for chunk in chat.reply("hello"): ...

Uses Ollama's native /api/chat (not its OpenAI-compatible endpoint) because
that is where `think: false` lives: reasoning models
would otherwise spend seconds thinking before the first word. Any `<think>`
block a model emits anyway is stripped from the stream.

Server: OLLAMA_HOST (default http://localhost:11434). Model: --model or
OLLAMA_MODEL; `ollama list` shows what is pulled.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Iterator, List, Optional

from .claude_chat import VISION_RULES, VOICE_RULES, WAKE_RULES, load_system_prompt

DEFAULT_HOST = "http://localhost:11434"


def list_models(host: Optional[str] = None) -> List[str]:
    host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
    with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
        return [m["name"] for m in json.load(r).get("models", [])]


class OllamaChat:
    name = "ollama"

    def __init__(self, model: Optional[str] = None, character: Optional[str] = None,
                 host: Optional[str] = None, max_history: int = 20, temperature: float = 0.8,
                 can_see: bool = False, wake_mode: bool = False, keep_alive: str = "30m"):
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL")
        if not self.model:
            try:
                have = ", ".join(list_models(self.host)[:8])
            except Exception:
                have = "(server not reachable)"
            raise RuntimeError(f"Ollama needs a model: --model <name> or OLLAMA_MODEL. Pulled: {have}")
        self.max_history = max_history
        self.temperature = temperature
        self.keep_alive = keep_alive          # keep the model loaded between turns
        self.system = (load_system_prompt() + VOICE_RULES + (VISION_RULES if can_see else "")
                       + (WAKE_RULES if wake_mode else ""))
        if character:
            self.system += f"\n\nCharacter: {character}"
        self.messages: List[dict] = []
        self.last_usage = None
        self._pending_context: Optional[str] = None

    # same context hooks as the other brains
    def add_context(self, text: str) -> None:
        self._pending_context = text.strip() or None

    def _push_user(self, user_text: str) -> None:
        if self._pending_context:
            self.messages.append({"role": "user", "content": f"(You notice: {self._pending_context})"})
            self._pending_context = None
        self.messages.append({"role": "user", "content": user_text})

    def warm_up(self) -> None:
        """Load the model into memory now so the first reply is not slow."""
        try:
            body = json.dumps({"model": self.model, "keep_alive": self.keep_alive}).encode()
            req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=300).read()
        except Exception:
            pass

    def reply(self, user_text: str) -> Iterator[str]:
        self._push_user(user_text)
        self.messages = self.messages[-self.max_history:]
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}] + self.messages,
            "stream": True,
            "think": False,                                   # no reasoning pass: fast first token
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": 256},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        parts: List[str] = []
        in_think = False
        try:
            try:
                resp = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"Ollama {e.code}: {e.read().decode(errors='replace')[:200]}")
            except urllib.error.URLError as e:
                raise RuntimeError(f"Ollama not reachable at {self.host}: {e.reason}")
            with resp:
                for raw in resp:
                    if not raw.strip():
                        continue
                    data = json.loads(raw)
                    if data.get("error"):
                        raise RuntimeError(f"Ollama: {data['error']}")
                    text = (data.get("message") or {}).get("content", "")
                    if text:
                        # strip any <think>...</think> a model emits despite think=false
                        if "<think>" in text:
                            in_think = True
                        if in_think:
                            if "</think>" in text:
                                in_think = False
                                text = text.split("</think>", 1)[1]
                            else:
                                text = ""
                        if text:
                            parts.append(text)
                            yield text
                    if data.get("done"):
                        self.last_usage = {"eval_count": data.get("eval_count"),
                                           "eval_duration_ms": round((data.get("eval_duration") or 0) / 1e6)}
                        break
        finally:
            full = re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.S).strip()
            if full:
                self.messages.append({"role": "assistant", "content": full})
            else:
                self.messages.pop()
