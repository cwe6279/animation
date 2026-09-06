"""
stt_backends.py
===============
Streaming speech-to-text for the voice round trip.

    stt = make_stt("vosk")                 # or "whisper"
    for pcm in mic_chunks:                 # int16 mono at stt.sample_rate
        t = stt.feed(pcm)
        if t and t.final: print("user said:", t.text)
        elif t: print("partial:", t.text)

Backends:
    VoskSTT     local, streaming, built-in end-of-utterance detection, partial
                results while the user is still talking (~40 MB model,
                downloaded to ~/.cache/talker on first use). Best latency.
    WhisperSTT  local faster-whisper; higher accuracy, but transcribes only
                after our energy-based endpointer decides the user stopped.

Both expose `speech_active` so the loop can interrupt playback on barge-in.
"""

from __future__ import annotations

import json
import os
import queue
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "talker")


@dataclass
class Transcript:
    text: str
    final: bool


class STTBackend:
    name = "base"
    sample_rate = 16000
    speech_active = False           # True while the user seems to be talking
    endpoint_delay_s = 0.0          # silence the endpointer waits for before deciding you stopped
    last_transcribe_s = 0.0         # time the last final transcription took (batch backends)

    def feed(self, pcm: bytes) -> Optional[Transcript]:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop buffered audio (e.g. after we discarded a stretch of playback echo)."""


# ─────────────────────────────────────────────────────
# ENERGY ENDPOINTER  (shared by Whisper; simple and predictable)
# ─────────────────────────────────────────────────────
class EnergyEndpointer:
    """
    Speech start/end from signal energy, robust to room noise:

      * The noise floor is tracked continuously (fast to fall, slow to rise),
        so a noisy mic or a fan does not read as speech; the first 0.3 s are
        used to calibrate it.
      * Speech starts when RMS exceeds max(min_rms, floor * start_ratio) on
        two consecutive frames; it ends after `silence_ms` below
        floor * end_ratio (hysteresis so trailing syllables don't cut off).
      * max_utterance_s is a hard cap.
    """

    def __init__(self, sample_rate: int, min_rms: float = 350.0, start_ratio: float = 3.0,
                 end_ratio: float = 1.8, silence_ms: int = 600, max_utterance_s: float = 15.0,
                 calibrate_s: float = 0.3):
        self.sample_rate = sample_rate
        self.min_rms = min_rms
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.max_samples = int(sample_rate * max_utterance_s)
        self.calibrate_samples = int(sample_rate * calibrate_s)
        self.floor = min_rms
        self.active = False
        self._calibrated = 0
        self._above = 0
        self._quiet = 0
        self._buf: list[bytes] = []
        self._n = 0
        self._pre: list[bytes] = []      # a little audio before onset so first syllables survive

    def _track_floor(self, rms: float) -> None:
        if self._calibrated < self.calibrate_samples:
            self.floor = rms if self._calibrated == 0 else 0.7 * self.floor + 0.3 * rms
            return
        if rms < self.floor:
            self.floor = 0.8 * self.floor + 0.2 * rms       # fall quickly
        else:
            self.floor = min(rms, self.floor * 1.01 + 1.0)  # rise slowly (~1 %/frame)

    def feed(self, pcm: bytes) -> Optional[bytes]:
        """Returns the utterance's audio when it ends, else None."""
        s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
        self._track_floor(rms)
        if self._calibrated < self.calibrate_samples:
            self._calibrated += s.size
            return None

        if not self.active:
            self._pre = (self._pre + [pcm])[-3:]
            start_thr = max(self.min_rms, self.floor * self.start_ratio)
            self._above = self._above + 1 if rms > start_thr else 0
            if self._above >= 2:
                self.active = True
                self._quiet = 0
                self._buf = list(self._pre)
                self._n = sum(len(b) // 2 for b in self._buf)
                self._above = 0
            return None

        self._buf.append(pcm)
        self._n += s.size
        end_thr = max(self.min_rms, self.floor * self.end_ratio)
        self._quiet = 0 if rms > end_thr else self._quiet + s.size
        if self._quiet >= self.silence_samples or self._n >= self.max_samples:
            audio = b"".join(self._buf)
            self.reset()
            return audio
        return None

    def reset(self) -> None:
        self.active = False
        self._above = 0
        self._quiet = 0
        self._buf = []
        self._n = 0


# ─────────────────────────────────────────────────────
# VOSK
# ─────────────────────────────────────────────────────
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"


VOSK_MODELS = {
    "small": "vosk-model-small-en-us-0.15",   # ~40 MB, fast, so-so accuracy
    "large": "vosk-model-en-us-0.22",         # ~1.8 GB, much better accuracy, ~1-2 GB RAM
    "lgraph": "vosk-model-en-us-0.22-lgraph", # ~130 MB, middle ground
}


def ensure_vosk_model(model_path: Optional[str] = None) -> str:
    """model_path may be a directory, a model name (vosk-model-...), or small|lgraph|large."""
    spec = model_path or os.environ.get("VOSK_MODEL") or "small"
    if os.path.isdir(spec):
        return spec
    name = VOSK_MODELS.get(spec, spec)
    path = os.path.join(CACHE_DIR, name)
    if os.path.isdir(path):
        return path
    if not name.startswith("vosk-model-"):
        raise FileNotFoundError(f"Vosk model not found: {spec}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = path + ".zip"
    print(f"[stt] downloading Vosk model {name} to {CACHE_DIR} (large one is ~1.8 GB) ...")
    urllib.request.urlretrieve(f"https://alphacephei.com/vosk/models/{name}.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CACHE_DIR)
    os.remove(zip_path)
    return path


class VoskSTT(STTBackend):
    name = "vosk"

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000,
                 silence_ms: Optional[int] = None):
        import vosk
        vosk.SetLogLevel(-1)
        self.sample_rate = sample_rate
        self._model = vosk.Model(ensure_vosk_model(model_path))
        self._rec = vosk.KaldiRecognizer(self._model, sample_rate)
        self._rec.SetWords(False)
        if silence_ms is not None:
            # Vosk's endpointer is rule based; scale its silence rules from the default 0.5 s.
            try:
                self._rec.SetEndpointerDelays(silence_ms / 1000.0, silence_ms / 1000.0 * 2, 30.0)
            except Exception:
                pass
        self.speech_active = False

    def feed(self, pcm: bytes) -> Optional[Transcript]:
        if not pcm:
            return None
        if self._rec.AcceptWaveform(pcm):
            text = json.loads(self._rec.Result()).get("text", "").strip()
            self.speech_active = False
            return Transcript(text, True) if text else None
        partial = json.loads(self._rec.PartialResult()).get("partial", "").strip()
        self.speech_active = bool(partial)
        return Transcript(partial, False) if partial else None

    def reset(self) -> None:
        self._rec.Reset()
        self.speech_active = False


# ─────────────────────────────────────────────────────
# FASTER-WHISPER
# ─────────────────────────────────────────────────────
_HALLUCINATIONS = {"you", "you.", "thank you.", "thank you", "thanks for watching.", "bye.", "the end.", "."}


def _looks_hallucinated(text: str, seconds: float) -> bool:
    """Short clip + one of Whisper's stock noise outputs = not real speech."""
    t = text.strip().lower()
    return (not t) or (seconds < 2.0 and t in _HALLUCINATIONS)

class WhisperSTT(STTBackend):
    name = "whisper"

    def __init__(self, model_size: str = "base.en", sample_rate: int = 16000,
                 device: str = "cpu", compute_type: str = "int8", silence_ms: int = 600):
        from faster_whisper import WhisperModel
        self.sample_rate = sample_rate
        print(f"[stt] loading faster-whisper {model_size} (endpoint after {silence_ms} ms of silence) ...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                   download_root=os.path.join(CACHE_DIR, "whisper"))
        self._ep = EnergyEndpointer(sample_rate, silence_ms=silence_ms)
        self.endpoint_delay_s = silence_ms / 1000.0
        self.speech_active = False

    def feed(self, pcm: bytes) -> Optional[Transcript]:
        audio = self._ep.feed(pcm)
        self.speech_active = self._ep.active
        if audio is None:
            return Transcript("", False) if self._ep.active else None
        t0 = time.monotonic()
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(samples, language="en", beam_size=1, vad_filter=False,
                                             condition_on_previous_text=False)
        kept = []
        for seg in segments:
            # Whisper invents "You", "Thank you." etc. on noise; drop low-confidence segments.
            if seg.no_speech_prob > 0.6 or seg.avg_logprob < -1.2:
                continue
            kept.append(seg.text.strip())
        text = " ".join(kept).strip()
        if _looks_hallucinated(text, len(samples) / self.sample_rate):
            text = ""
        self.last_transcribe_s = time.monotonic() - t0
        silence = self._ep.silence_samples / self.sample_rate
        print(f"[stt] whisper {len(samples)/self.sample_rate:.1f}s audio in {(time.monotonic()-t0)*1000:.0f} ms "
              f"(+{silence*1000:.0f} ms waiting for you to stop)")
        return Transcript(text, True) if text else None

    def reset(self) -> None:
        self._ep.reset()
        self.speech_active = False


# ─────────────────────────────────────────────────────
# ELEVENLABS SCRIBE (realtime websocket, server-side VAD)
# ─────────────────────────────────────────────────────
class ElevenLabsSTT(STTBackend):
    """
    Cloud speech-to-text over ElevenLabs' realtime Scribe websocket. The
    server does the endpointing (commit_strategy=vad): partial transcripts
    stream while you talk and a committed one arrives ~0.5 s after you stop.
    No local CPU, so it is the option for a Raspberry Pi. Needs
    ELEVENLABS_API_KEY with the speech-to-text permission.
    """
    name = "elevenlabs"
    MODEL = "scribe_v2_realtime"

    def __init__(self, sample_rate: int = 16000, silence_ms: Optional[int] = None,
                 api_key: Optional[str] = None, language: Optional[str] = None):
        import asyncio
        import threading
        self.sample_rate = sample_rate
        self.silence_s = (silence_ms or 500) / 1000.0
        self.endpoint_delay_s = self.silence_s
        self.language = language
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ElevenLabsSTT needs ELEVENLABS_API_KEY in the environment.")
        self.speech_active = False
        self._results: "queue.Queue[Transcript]" = queue.Queue()
        self._audio: Optional[asyncio.Queue] = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="scribe-ws", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _url(self) -> str:
        url = (f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id={self.MODEL}"
               f"&audio_format=pcm_{self.sample_rate}&commit_strategy=vad"
               f"&vad_silence_threshold_secs={self.silence_s}")
        if self.language:
            url += f"&language_code={self.language}"
        return url

    def _run(self) -> None:
        import asyncio
        asyncio.set_event_loop(self._loop)
        self._audio = asyncio.Queue()
        self._ready.set()
        self._loop.run_until_complete(self._session_forever())

    async def _session_forever(self) -> None:
        import asyncio
        import base64
        import aiohttp
        backoff = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.ws_connect(self._url(), headers={"xi-api-key": self.api_key},
                                               heartbeat=20) as ws:
                        print("[stt] scribe realtime connected")
                        backoff = 1.0

                        async def sender():
                            while True:
                                pcm = await self._audio.get()
                                await ws.send_json({"message_type": "input_audio_chunk",
                                                    "audio_base_64": base64.b64encode(pcm).decode(),
                                                    "commit": False, "sample_rate": self.sample_rate})

                        task = asyncio.ensure_future(sender())
                        try:
                            async for msg in ws:
                                if msg.type != aiohttp.WSMsgType.TEXT:
                                    break
                                d = json.loads(msg.data)
                                kind = d.get("message_type")
                                text = (d.get("text") or "").strip()
                                if kind == "partial_transcript":
                                    self.speech_active = bool(text)
                                    if text:
                                        self._results.put(Transcript(text, False))
                                elif kind == "committed_transcript":
                                    self.speech_active = False
                                    if text:
                                        self._results.put(Transcript(text, True))
                                elif kind and kind.endswith("error"):
                                    print(f"[stt] scribe: {d}")
                        finally:
                            task.cancel()
            except Exception as e:
                print(f"[stt] scribe connection lost ({e.__class__.__name__}: {str(e)[:120]}); "
                      f"reconnecting in {backoff:.0f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def feed(self, pcm: bytes) -> Optional[Transcript]:
        if pcm and self._audio is not None:
            self._loop.call_soon_threadsafe(self._audio.put_nowait, pcm)
        latest: Optional[Transcript] = None
        while not self._results.empty():
            t = self._results.get_nowait()
            if t.final:
                return t                 # deliver finals immediately
            latest = t
        return latest

    def reset(self) -> None:
        while not self._results.empty():
            self._results.get_nowait()
        self.speech_active = False


# ─────────────────────────────────────────────────────
# OPENAI-COMPATIBLE BATCH (Groq whisper-large-v3-turbo, OpenAI whisper-1)
# ─────────────────────────────────────────────────────
class OpenAICompatSTT(STTBackend):
    """
    Batch cloud transcription: our energy endpointer decides when you stopped,
    then the clip is uploaded as a WAV. No partials. Groq is fast and cheap
    (Whisper large on their hardware); OpenAI is the reference.
    """
    name = "openai"

    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str] = None,
                 sample_rate: int = 16000, silence_ms: Optional[int] = None, name: str = "openai",
                 client=None):
        self.name = name
        self.model = model
        self.sample_rate = sample_rate
        silence_ms = silence_ms or 600
        self._ep = EnergyEndpointer(sample_rate, silence_ms=silence_ms)
        self.endpoint_delay_s = silence_ms / 1000.0
        self.speech_active = False
        if client is None:
            from openai import OpenAI
            if not api_key:
                raise RuntimeError(f"{name} STT needs an API key in the environment")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    @classmethod
    def groq(cls, model: Optional[str] = None, **kw):
        return cls(model or "whisper-large-v3-turbo", os.environ.get("GROQ_API_KEY"),
                   "https://api.groq.com/openai/v1", name="groq", **kw)

    @classmethod
    def openai(cls, model: Optional[str] = None, **kw):
        return cls(model or "whisper-1", os.environ.get("OPENAI_API_KEY"), None, name="openai", **kw)

    def feed(self, pcm: bytes) -> Optional[Transcript]:
        audio = self._ep.feed(pcm)
        self.speech_active = self._ep.active
        if audio is None:
            return Transcript("", False) if self._ep.active else None
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(audio)
        buf.seek(0)
        buf.name = "utterance.wav"
        t0 = time.monotonic()
        try:
            result = self._client.audio.transcriptions.create(
                model=self.model, file=buf, language="en", response_format="text")
        except Exception as e:
            print(f"[stt] {self.name} transcription failed: {e}")
            return None
        text = (result if isinstance(result, str) else getattr(result, "text", "")).strip()
        self.last_transcribe_s = time.monotonic() - t0
        print(f"[stt] {self.name} {len(audio)/2/self.sample_rate:.1f}s audio in "
              f"{(time.monotonic()-t0)*1000:.0f} ms (+{self.endpoint_delay_s*1000:.0f} ms waiting for you to stop)")
        return Transcript(text, True) if text else None

    def reset(self) -> None:
        self._ep.reset()
        self.speech_active = False


STT_BACKENDS = {"vosk": VoskSTT, "whisper": WhisperSTT, "elevenlabs": ElevenLabsSTT,
                "groq": OpenAICompatSTT.groq, "openai": OpenAICompatSTT.openai}


def make_stt(name: str = "vosk", **kwargs) -> STTBackend:
    try:
        cls = STT_BACKENDS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown STT backend {name!r}; choose from {sorted(STT_BACKENDS)}")
    return cls(**{k: v for k, v in kwargs.items() if v is not None})
