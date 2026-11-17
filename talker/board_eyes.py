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


DOCUMENT_PROMPT = (
    "This photo shows a document, page, label or screen. Transcribe all readable text exactly as "
    "written, keeping its line breaks, headings and lists; mark anything unreadable as [unclear]. "
    "Do not add or correct anything. Then one last line starting SUMMARY: saying what the document is."
)


def document_views(jpeg: bytes, max_pixels: int = 1_150_000) -> list:
    """For a scan: the whole page, then its four quarters (overlapping 6 % each way). The vision
    models shrink an image to about 1.15 megapixels (Claude: also 1568 px on the long side), so a
    2048x1536 photo reaches them at about 60 % and small print turns to mush; a quarter, about
    1100x830, fits whole and keeps every pixel. Photos already small enough go as they are."""
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[0] * img.shape[1] <= max_pixels:
            return [jpeg]
        h, w = img.shape[:2]
        dy, dx = int(h * 0.06), int(w * 0.06)
        quarters = [img[:h // 2 + dy, :w // 2 + dx], img[:h // 2 + dy, w // 2 - dx:],
                    img[h // 2 - dy:, :w // 2 + dx], img[h // 2 - dy:, w // 2 - dx:]]
        return [jpeg] + [cv2.imencode(".jpg", q, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for q in quarters]
    except Exception:
        return [jpeg]


DOCUMENT_VIEWS_NOTE = ("The first image is the whole page; the next four are its quarters at full detail (top "
                       "left, top right, bottom left, bottom right; they overlap a little): read the text from the "
                       "quarters, in reading order across the page, and do not repeat the overlapping words.\n")


def _prompt(question: str, mode: str = "scene", views: int = 1) -> str:
    if mode == "document":
        return (DOCUMENT_VIEWS_NOTE if views > 1 else "") + DOCUMENT_PROMPT
    q = (question or "").strip()
    return PHOTO_PROMPT + (f"\nThe person asked: {q}\nAnswer that too, on a last line starting ANSWER:." if q else "")


def make_photo_describer(backend: str = "anthropic", claude_model: str = "claude-haiku-4-5",
                         ollama_url: str = "http://localhost:11434", ollama_model: str = "gemma4:31b",
                         timeout: float = 60.0) -> Callable[..., str]:
    """describe(jpeg, question, mode="scene"|"document") -> text, for --vision-backend 'anthropic' or 'ollama' ('auto' tries
    the local model first). Claude uses ANTHROPIC_VISION_API_KEY when set, as talker's vision does."""
    client = {"c": None}

    def claude(jpeg: bytes, question: str, mode: str = "scene") -> str:
        if client["c"] is None:
            key = os.environ.get("ANTHROPIC_VISION_API_KEY", "").strip()
            if key:
                import anthropic
                client["c"] = anthropic.Anthropic(api_key=key)
            else:
                from .brains.claude_chat import make_client
                client["c"] = make_client()
        images = document_views(jpeg) if mode == "document" else [jpeg]
        resp = client["c"].messages.create(
            model=claude_model, max_tokens=1500 if mode == "document" else 500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(im).decode()}} for im in images
            ] + [{"type": "text", "text": _prompt(question, mode, len(images))}]}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def local(jpeg: bytes, question: str, mode: str = "scene") -> str:
        images = document_views(jpeg) if mode == "document" else [jpeg]
        body = {"model": ollama_model, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0.2},
                "messages": [{"role": "user", "content": _prompt(question, mode, len(images)),
                              "images": [base64.b64encode(im).decode() for im in images]}]}
        req = urllib.request.Request(ollama_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return ((json.load(r).get("message") or {}).get("content") or "").strip()

    if backend == "ollama":
        return local
    if backend == "auto":
        def auto(jpeg: bytes, question: str, mode: str = "scene") -> str:
            try:
                return local(jpeg, question, mode)
            except Exception as e:
                print(f"[eyes] local model unreachable ({str(e)[:80]}); using Claude")
                return claude(jpeg, question, mode)
        return auto
    return claude


WECHAT_QR_DIR = os.environ.get("WECHAT_QR_DIR", os.path.expanduser("~/.cache/talker/wechat_qr"))
_wechat = {"det": None, "tried": False}


def _wechat_detector():
    """OpenCV's WeChat QR detector (opencv-contrib): far better at small, blurred or tilted codes.
    With its two small CNN models in WECHAT_QR_DIR (detect.* and sr.*) it finds codes better
    still; without them it falls back to its classic finder."""
    if not _wechat["tried"]:
        _wechat["tried"] = True
        try:
            import cv2
            files = [os.path.join(WECHAT_QR_DIR, f) for f in
                     ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")]
            _wechat["det"] = (cv2.wechat_qrcode_WeChatQRCode(*files) if all(os.path.exists(f) for f in files)
                              else cv2.wechat_qrcode_WeChatQRCode())
        except Exception:
            _wechat["det"] = None
    return _wechat["det"]


def read_qr_codes(jpeg: bytes) -> list:
    """The text of every QR code in the photo, read on the host (free, tens of ms at 1600x1200).
    A vision model can see that a QR code is there but cannot decode it."""
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        found = []
        wechat = _wechat_detector()
        if wechat is not None:
            texts, _ = wechat.detectAndDecode(img)
            found = [t for t in texts if t]
        if not found:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            det = cv2.QRCodeDetector()
            for frame in (grey, cv2.resize(grey, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)):
                ok, texts, _, _ = det.detectAndDecodeMulti(frame)
                found = [t for t in texts if t] if ok else []
                if found:
                    break
        return list(dict.fromkeys(found))          # unique, in order
    except Exception as e:
        print(f"[eyes] QR reading failed: {e}")
        return []


class BoardEyes:
    """What the voice loop calls `vision` when the board is the camera: look_now() asks the board
    for a photo and returns its description ("" when no board is connected or it takes too long)."""

    on_demand = True

    def __init__(self, server, describe: Callable[..., str], timeout: float = 15.0,
                 on_error: Optional[Callable[[str], None]] = None, save_dir: Optional[str] = None):
        self.server = server
        self.describe = describe
        self.timeout = timeout
        self.on_error = on_error or (lambda msg: print(f"[eyes] {msg}"))
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._asked = 0          # photos requested
        self._answered = 0       # photos described (or failed)
        self._latest = ""
        self._mode = "scene"       # the pending request's mode: "scene" or "document"
        self._save = ""            # the pending request's file prefix ("photo", "scan"), "" = not kept
        self.save_dir = save_dir   # where "take a photo" and "scan this" keep their files
        self.last_saved: Optional[str] = None
        self.last_jpeg: Optional[bytes] = None
        self.stats = {"described": 0, "failed": 0, "last_ms": 0}
        server.photo_handler = self._on_photo
        server.photo_failed = self._on_failed

    # -- called by the satellite server (its thread) --------------------------------
    def _on_photo(self, jpeg: bytes, question: str) -> str:
        t0 = time.monotonic()
        with self._lock:
            mode, save = self._mode, self._save
        try:
            text = self.describe(jpeg, question, mode) if mode != "scene" else self.describe(jpeg, question)
            self.stats["described"] += 1
        except Exception as e:
            text = ""
            self.stats["failed"] += 1
            self.on_error(f"describing the photo failed: {str(e)[:160]}")
        codes = read_qr_codes(jpeg)
        if codes:
            print(f"[eyes] QR code: {' | '.join(c[:80] for c in codes)}")
            text = (text + "\n" if text else "") + "\n".join(f"QR CODE (decoded exactly): {c}" for c in codes)
        self.stats["last_ms"] = round((time.monotonic() - t0) * 1000)
        if save and self.save_dir:
            self.last_saved = self._keep(jpeg, text, save)
        elif self.save_dir:                      # the latest look, overwritten each time: what did she get?
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                with open(os.path.join(self.save_dir, "last-look.jpg"), "wb") as f:
                    f.write(jpeg)
            except OSError:
                pass
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
    def request(self, force: bool = True) -> None:
        """The voice loop asks for a look whenever someone starts speaking. For the board's
        camera that is not a reason to take a photo (as for an on-demand local camera): photos
        are taken on {{look}} and on visual questions only."""
        return None

    def _keep(self, jpeg: bytes, text: str, prefix: str) -> Optional[str]:
        """Save the photo (and its description or transcription) as <prefix>-<time>.jpg/.txt."""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            base = os.path.join(self.save_dir, f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}")
            with open(base + ".jpg", "wb") as f:
                f.write(jpeg)
            if text:
                with open(base + ".txt", "w") as f:
                    f.write(text.strip() + "\n")
            print(f"[eyes] saved {base}.jpg" + (" and .txt" if text else ""))
            return base + ".jpg"
        except OSError as e:
            self.on_error(f"could not save the photo: {e}")
            return None

    def look_now(self, timeout: Optional[float] = None, question: str = "", mode: str = "scene",
                 save: str = "") -> str:
        """A fresh photo's description (mode "document": its text, transcribed). The voice loop's
        own timeout (meant for a local camera that is already running) is too short for capture +
        upload + description, so the longer of the two applies. save: a file prefix to keep the
        photo and its text in save_dir ("photo", "scan"); "" keeps nothing."""
        wait = max(timeout or 0.0, self.timeout + (20.0 if mode == "document" else 0.0))
        with self._done:
            self._asked += 1
            ticket = self._asked
            self._mode, self._save = mode, save
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

    def pending(self) -> bool:
        """A photo has been asked for and not yet described: keep the board's channel open."""
        with self._lock:
            return self._answered < self._asked

    def latest(self) -> str:
        return self._latest
