"""Eyes on the satellite board: photos from the board's own camera, described on the host.

The board (xiaozhi firmware) has a `self.camera.take_photo` tool. The host gives it an upload
address in the MCP handshake; when the host calls the tool, the board takes one JPEG and posts
it there with a question, and whatever text the host answers becomes the tool's result. Here
the host describes the photo itself, so the description also comes straight back to the brain
without waiting for the tool's reply.

The prompt asks for an itemized list (people, what is held, objects, readable text, setting):
a two-sentence prompt drops most of what is in view. Measured on a 1600x1200 photo: about 4 s
on a local Gemma 4 31B with thinking off; Claude Haiku is similar.
"""
import base64
import json
import os
import threading
import time
import urllib.request
from typing import Callable, Optional

PHOTO_PROMPT = (
    "You are the eyes of a small companion device; this is one photo from its camera. "
    "Report everything useful you can see, as short plain lines:\n"
    "PEOPLE: each person, how they look, what they wear, expression, what they are doing.\n"
    "HELD: anything in someone's hands, as specifically as you can (colour, kind of object, brand or text).\n"
    "OBJECTS: every other object you can make out, with where it is.\n"
    "TEXT: any readable writing, quoted exactly, or 'none'.\n"
    "SETTING: the room, lighting, camera angle.\n"
    "Only what is actually visible; say 'unclear' instead of guessing."
)


def _prompt(question: str) -> str:
    q = (question or "").strip()
    return PHOTO_PROMPT + (f"\nThe person asked: {q}\nAnswer that too, on a last line starting ANSWER:." if q else "")


def make_photo_describer(backend: str = "anthropic", claude_model: str = "claude-haiku-4-5",
                         ollama_url: str = "http://localhost:11434", ollama_model: str = "gemma4:31b",
                         timeout: float = 60.0) -> Callable[[bytes, str], str]:
    """describe(jpeg, question) -> text, for --vision-backend 'anthropic' or 'ollama' ('auto' tries
    the local model first). Claude uses ANTHROPIC_VISION_API_KEY when set, as talker's vision does."""
    client = {"c": None}

    def claude(jpeg: bytes, question: str) -> str:
        if client["c"] is None:
            key = os.environ.get("ANTHROPIC_VISION_API_KEY", "").strip()
            if key:
                import anthropic
                client["c"] = anthropic.Anthropic(api_key=key)
            else:
                from .brains.claude_chat import make_client
                client["c"] = make_client()
        resp = client["c"].messages.create(
            model=claude_model, max_tokens=500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": _prompt(question)}]}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def local(jpeg: bytes, question: str) -> str:
        body = {"model": ollama_model, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0.2},
                "messages": [{"role": "user", "content": _prompt(question),
                              "images": [base64.b64encode(jpeg).decode()]}]}
        req = urllib.request.Request(ollama_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return ((json.load(r).get("message") or {}).get("content") or "").strip()

    if backend == "ollama":
        return local
    if backend == "auto":
        def auto(jpeg: bytes, question: str) -> str:
            try:
                return local(jpeg, question)
            except Exception as e:
                print(f"[eyes] local model unreachable ({str(e)[:80]}); using Claude")
                return claude(jpeg, question)
        return auto
    return claude


class BoardEyes:
    """What the voice loop calls `vision` when the board is the camera: look_now() asks the board
    for a photo and returns its description ("" when no board is connected or it takes too long)."""

    on_demand = True

    def __init__(self, server, describe: Callable[[bytes, str], str], timeout: float = 15.0,
                 on_error: Optional[Callable[[str], None]] = None):
        self.server = server
        self.describe = describe
        self.timeout = timeout
        self.on_error = on_error or (lambda msg: print(f"[eyes] {msg}"))
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._asked = 0          # photos requested
        self._answered = 0       # photos described (or failed)
        self._latest = ""
        self.last_jpeg: Optional[bytes] = None
        self.stats = {"described": 0, "failed": 0, "last_ms": 0}
        server.photo_handler = self._on_photo
        server.photo_failed = self._on_failed

    # -- called by the satellite server (its thread) --------------------------------
    def _on_photo(self, jpeg: bytes, question: str) -> str:
        t0 = time.monotonic()
        try:
            text = self.describe(jpeg, question)
            self.stats["described"] += 1
        except Exception as e:
            text = ""
            self.stats["failed"] += 1
            self.on_error(f"describing the photo failed: {str(e)[:160]}")
        self.stats["last_ms"] = round((time.monotonic() - t0) * 1000)
        with self._done:
            self.last_jpeg = jpeg
            if text:
                self._latest = text
            self._answered = self._asked
            self._done.notify_all()
        print(f"[eyes] photo {len(jpeg) // 1024} KB described in {self.stats['last_ms']} ms")
        return text or "I could not make out the photo."

    def _on_failed(self, reason: str) -> None:
        self.on_error(f"the board could not take a photo: {reason[:160]}")
        with self._done:
            self.stats["failed"] += 1
            self._answered = self._asked
            self._done.notify_all()

    # -- called by the voice loop ---------------------------------------------------
    def look_now(self, timeout: Optional[float] = None, question: str = "") -> str:
        """A fresh photo's description. The voice loop's own timeout (meant for a local camera
        that is already running) is too short for capture + upload + description, so the
        longer of the two applies."""
        wait = max(timeout or 0.0, self.timeout)
        with self._done:
            self._asked += 1
            ticket = self._asked
        if not self.server.take_photo(question):
            with self._done:
                self._answered = max(self._answered, ticket)
            return ""
        deadline = time.monotonic() + wait
        with self._done:
            while self._answered < ticket:
                left = deadline - time.monotonic()
                if left <= 0:
                    self.on_error(f"no photo from the board within {wait:.0f} s")
                    return ""
                self._done.wait(left)
            return self._latest

    def latest(self) -> str:
        return self._latest
