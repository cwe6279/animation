"""
tts_backends.py
===============
Streaming text-to-speech backends.

A backend is an async generator: it consumes sentences as they become
available and yields TTSEvents as soon as it has them, so playback can start
on the first sentence while later ones are still being synthesized.

    async for ev in backend.synthesize(sentence_queue):
        AudioChunk(pcm)             16-bit mono PCM at backend.sample_rate
        WordBoundary(word, t0, t1)  seconds relative to the first audio sample
                                    of this synthesize() session
        SentenceDone(text)          bookkeeping for the pipeline

Backends:
    EdgeTTSBackend        free, no key, ~300 ms to first audio, word timings.
                          Only outputs MP3, so it is decoded through a
                          long-lived ffmpeg pipe (~50-100 ms extra).
    ElevenLabsBackend     paid, ~100-200 ms to first audio with Flash models,
                          raw PCM (no decode step), character-level timings.
                          Needs ELEVENLABS_API_KEY (.env is loaded by the apps).

Select with `make_backend("edge")` / `make_backend("elevenlabs", voice=...)`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────
# EVENTS
# ─────────────────────────────────────────────────────
@dataclass
class AudioChunk:
    pcm: bytes                 # int16 mono little-endian


@dataclass
class WordBoundary:
    word: str
    start: float               # seconds from the session's first audio sample
    end: float


@dataclass
class SentenceDone:
    text: str


TTSEvent = Union[AudioChunk, WordBoundary, SentenceDone]

# Sentinel pushed onto the sentence queue when the utterance is complete.
END_OF_TEXT = None


class TTSBackend:
    """Base class. Subclasses set sample_rate and implement synthesize()."""
    name = "base"
    sample_rate = 24000
    supports_audio_tags = False   # can the voice perform "[sigh]", "[excited]" etc.?

    async def synthesize(self, sentences: "asyncio.Queue[Optional[str]]") -> AsyncIterator[TTSEvent]:
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator)

    async def warm_up(self) -> None:
        """Optional: pre-open connections / spawn helpers to cut first-utterance latency."""
        return None

    async def close(self) -> None:
        """Release connections. Called when the speech pipeline stops."""
        return None


# ─────────────────────────────────────────────────────
# FFMPEG PIPE DECODER  (MP3 stream -> PCM stream)
# ─────────────────────────────────────────────────────
def find_ffmpeg() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


class FfmpegPipeDecoder:
    """
    One ffmpeg process that decodes an MP3 byte stream to s16le PCM as the
    bytes arrive. Feed compressed bytes with write(); PCM arrives on an
    asyncio queue via a reader thread. close() signals EOF; ffmpeg flushes
    its last frames and the reader thread pushes None.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, sample_rate: int,
                 ffmpeg_path: Optional[str] = None):
        ffmpeg = ffmpeg_path or find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found. Install it (brew/apt install ffmpeg) or "
                "`pip install imageio-ffmpeg` for a bundled binary.")
        self._proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-probesize", "32", "-analyzeduration", "0",
             "-fflags", "nobuffer", "-flags", "low_delay",
             "-f", "mp3", "-i", "pipe:0",
             "-f", "s16le", "-acodec", "pcm_s16le",
             "-ar", str(sample_rate), "-ac", "1",
             "-flush_packets", "1", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._loop = loop
        self.pcm_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        out = self._proc.stdout
        try:
            while True:
                data = out.read(4096)
                if not data:
                    break
                self._loop.call_soon_threadsafe(self.pcm_queue.put_nowait, data)
        finally:
            self._loop.call_soon_threadsafe(self.pcm_queue.put_nowait, None)

    def write(self, data: bytes) -> None:
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # ffmpeg died; stderr is reported by close()

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except Exception:
            pass

    def kill(self) -> None:
        self.close()
        try:
            self._proc.kill()
        except Exception:
            pass

    def error_text(self) -> str:
        try:
            self._proc.wait(timeout=2)
            return (self._proc.stderr.read() or b"").decode(errors="replace").strip()
        except Exception:
            return ""


# ─────────────────────────────────────────────────────
# EDGE-TTS BACKEND
# ─────────────────────────────────────────────────────
class EdgeTTSBackend(TTSBackend):
    name = "edge"
    sample_rate = 24000   # edge-tts serves 24 kHz mono MP3

    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+0%",
                 pitch: str = "+0Hz", speed: Optional[float] = None):
        self.voice = voice
        self.rate = f"{round((speed - 1.0) * 100):+d}%" if speed else rate
        self.pitch = pitch
        if find_ffmpeg() is None:
            raise RuntimeError("EdgeTTSBackend needs ffmpeg to decode MP3 "
                               "(install ffmpeg or `pip install imageio-ffmpeg`).")

    async def synthesize(self, sentences):
        import edge_tts
        loop = asyncio.get_running_loop()
        session_frames = 0      # PCM frames emitted so far in this session

        while True:
            sentence = await sentences.get()
            if sentence is END_OF_TEXT:
                return
            if not sentence.strip():
                continue

            base_t = session_frames / self.sample_rate
            decoder = FfmpegPipeDecoder(loop, self.sample_rate)
            comm = edge_tts.Communicate(sentence, voice=self.voice, rate=self.rate,
                                        pitch=self.pitch, boundary="WordBoundary")
            words: List[WordBoundary] = []
            words_emitted = 0

            async def pump_mp3():
                try:
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            decoder.write(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            t0 = chunk["offset"] / 1e7        # 100 ns units
                            t1 = t0 + chunk["duration"] / 1e7
                            words.append(WordBoundary(chunk["text"], base_t + t0, base_t + t1))
                finally:
                    decoder.close()

            pump = asyncio.ensure_future(pump_mp3())
            try:
                while True:
                    pcm = await decoder.pcm_queue.get()
                    if pcm is None:
                        break
                    # Yield word timings before the audio that carries them so
                    # the pipeline has the schedule ready when the sound plays.
                    while words_emitted < len(words):
                        yield words[words_emitted]
                        words_emitted += 1
                    session_frames += len(pcm) // 2
                    yield AudioChunk(pcm)
                await pump
                while words_emitted < len(words):
                    yield words[words_emitted]
                    words_emitted += 1
                if session_frames / self.sample_rate <= base_t:
                    err = decoder.error_text()
                    raise RuntimeError(f"edge-tts produced no audio for {sentence!r}"
                                       + (f": {err}" if err else ""))
                yield SentenceDone(sentence)
            finally:
                if not pump.done():
                    pump.cancel()
                decoder.kill()


# ─────────────────────────────────────────────────────
# ELEVENLABS BACKEND  (Flash/Turbo: websocket; v3: HTTP stream) — char alignment
# ─────────────────────────────────────────────────────
class _AlignmentWordizer:
    """
    Groups ElevenLabs character alignment into words, keeping state across
    the chunks a response arrives in (a word or a [tag] can straddle two).
    Handles both shapes:
      websocket: chars / charStartTimesMs / charDurationsMs
      HTTP:      characters / character_start_times_seconds / character_end_times_seconds
    Bracketed performance tags are in the alignment but are not spoken words,
    so they are skipped.
    """

    def __init__(self, base_t: float):
        self.base_t = base_t
        self._cur = ""
        self._t0 = self._t1 = 0.0
        self._in_tag = False

    def feed(self, al: dict) -> List[WordBoundary]:
        chars = al.get("chars") or al.get("characters") or []
        if "character_start_times_seconds" in al:
            starts = al["character_start_times_seconds"]
            ends = al.get("character_end_times_seconds") or starts
        else:
            starts_ms = al.get("charStartTimesMs") or al.get("char_start_times_ms") or []
            durs_ms = (al.get("charDurationsMs") or al.get("charsDurationsMs")
                       or al.get("char_durations_ms") or [0] * len(starts_ms))
            starts = [x / 1000.0 for x in starts_ms]
            ends = [(x + d) / 1000.0 for x, d in zip(starts_ms, durs_ms)]
        out: List[WordBoundary] = []
        for ch, s, e in zip(chars, starts, ends):
            if ch == "[":
                self._in_tag = True
                continue
            if self._in_tag:
                if ch == "]":
                    self._in_tag = False
                continue
            if ch.isspace():
                out.extend(self.flush())
                continue
            if not self._cur:
                self._t0 = s
            self._cur += ch
            self._t1 = e
        return out

    def flush(self) -> List[WordBoundary]:
        if not self._cur:
            return []
        wb = WordBoundary(self._cur, self.base_t + self._t0, self.base_t + self._t1)
        self._cur = ""
        return [wb]


MODEL_ALIASES = {
    "flash": "eleven_flash_v2_5",     # ~0.25 s to first audio; tags stripped
    "turbo": "eleven_turbo_v2_5",
    "v3": "eleven_v3",                # performs [tags]; ~1 s to first audio
    "multilingual": "eleven_multilingual_v2",
}


class ElevenLabsBackend(TTSBackend):
    """
    Uses the multi-context-free "stream-input" websocket:
      wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
    One connection per synthesize() session; each sentence is sent with
    flush=true so audio for it is generated immediately.

    Alignment arrives per audio message as parallel char arrays with start
    times relative to that message's audio; we accumulate an offset from the
    PCM byte count so WordBoundary times are session-relative.

    Verified live (Sept 2026): messages carry "audio" (base64 PCM) plus
    "alignment" {chars, charStartTimesMs, charDurationsMs}; with flush=true a
    whole sentence arrives as one audio message.
    """
    name = "elevenlabs"
    sample_rate = 24000

    def __init__(self, voice: str = "21m00Tcm4TlvDq8ikWAM", model: str = "eleven_flash_v2_5",
                 api_key: Optional[str] = None, stability: float = 0.5,
                 similarity: float = 0.75, speed: Optional[float] = None):
        self.voice_id = voice
        self.model = MODEL_ALIASES.get(model.lower(), model)
        self.speed = speed          # 0.7-1.2; honoured by Flash/Turbo, ignored by v3
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ElevenLabsBackend needs ELEVENLABS_API_KEY in the environment.")
        self.stability = stability
        self.similarity = similarity
        # Eleven v3 performs audio tags but is not offered on the websocket
        # (403), so it goes through the HTTP streaming endpoint with timestamps,
        # one request per sentence. Flash/Turbo use the websocket and get tags
        # stripped (they would read them aloud).
        self.supports_audio_tags = self.model.startswith("eleven_v3")
        self.transport = "http" if self.supports_audio_tags else "ws"

    def _voice_settings(self) -> dict:
        vs = {"stability": self.stability, "similarity_boost": self.similarity}
        if self.speed:
            vs["speed"] = max(0.7, min(1.2, float(self.speed)))
        return vs

    def _url(self) -> str:
        return (f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
                f"?model_id={self.model}&output_format=pcm_{self.sample_rate}")

    @staticmethod
    def _alignment_words(al: dict, base_t: float) -> List[WordBoundary]:
        """One-shot helper (tests): group a complete alignment into words."""
        w = _AlignmentWordizer(base_t)
        return w.feed(al) + w.flush()

    async def synthesize(self, sentences):
        if self.transport == "http":
            async for ev in self._synthesize_http(sentences):
                yield ev
        else:
            async for ev in self._synthesize_ws(sentences):
                yield ev

    async def _http_session(self):
        """One keep-alive HTTP session per backend: no TLS handshake per reply."""
        import aiohttp
        if getattr(self, "_http", None) is None or self._http.closed:
            self._http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(keepalive_timeout=120))
        return self._http

    async def warm_up(self) -> None:
        if self.transport == "http":
            try:
                http = await self._http_session()
                async with http.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": self.api_key}):
                    pass          # establishes the TLS connection; the response itself is irrelevant
            except Exception:
                pass

    async def close(self) -> None:
        http = getattr(self, "_http", None)
        if http is not None and not http.closed:
            await http.close()

    async def _synthesize_http(self, sentences):
        """One POST .../stream/with-timestamps per sentence (NDJSON lines)."""
        session_frames = 0
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
               f"/stream/with-timestamps?output_format=pcm_{self.sample_rate}")
        http = await self._http_session()
        if True:
            while True:
                sentence = await sentences.get()
                if sentence is END_OF_TEXT:
                    return
                if not sentence.strip():
                    continue
                base_t = session_frames / self.sample_rate
                wordizer = _AlignmentWordizer(base_t)
                body = {"text": sentence, "model_id": self.model,
                        "voice_settings": self._voice_settings()}
                async with http.post(url, headers={"xi-api-key": self.api_key}, json=body) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"ElevenLabs HTTP {resp.status}: {(await resp.text())[:300]}")
                    buf = b""
                    async for chunk in resp.content.iter_any():
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            al = data.get("alignment") or data.get("normalized_alignment")
                            if al:
                                for wb in wordizer.feed(al):
                                    yield wb
                            if data.get("audio_base64"):
                                pcm = base64.b64decode(data["audio_base64"])
                                session_frames += len(pcm) // 2
                                yield AudioChunk(pcm)
                for wb in wordizer.flush():
                    yield wb
                yield SentenceDone(sentence)

    async def _synthesize_ws(self, sentences):
        import aiohttp   # already a dependency of edge-tts
        session_frames = 0
        pending: List[str] = []

        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(self._url(), headers={"xi-api-key": self.api_key},
                                       heartbeat=20) as ws:
                await ws.send_json({
                    "text": " ",
                    "voice_settings": self._voice_settings(),
                    "generation_config": {"chunk_length_schedule": [50, 90, 120, 150]},
                })

                async def sender():
                    while True:
                        s = await sentences.get()
                        if s is END_OF_TEXT:
                            await ws.send_json({"text": ""})   # close the input side
                            return
                        if s.strip():
                            pending.append(s)
                            await ws.send_json({"text": s.strip() + " ", "flush": True})

                send_task = asyncio.ensure_future(sender())
                wordizer = _AlignmentWordizer(0.0)
                try:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                            continue
                        data = json.loads(msg.data)
                        if data.get("error") or data.get("message") and data.get("code"):
                            raise RuntimeError(f"ElevenLabs: {data}")
                        wordizer.base_t = session_frames / self.sample_rate
                        al = data.get("alignment") or data.get("normalizedAlignment")
                        if al:
                            for wb in wordizer.feed(al):
                                yield wb
                        if data.get("audio"):
                            pcm = base64.b64decode(data["audio"])
                            session_frames += len(pcm) // 2
                            yield AudioChunk(pcm)
                        if data.get("isFinal"):
                            break
                    for wb in wordizer.flush():
                        yield wb
                finally:
                    if not send_task.done():
                        send_task.cancel()
        for s in pending:
            yield SentenceDone(s)


# ─────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────
BACKENDS = {
    "edge": EdgeTTSBackend,
    "elevenlabs": ElevenLabsBackend,
}


def make_backend(name: str = "edge", **kwargs) -> TTSBackend:
    try:
        cls = BACKENDS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown TTS backend {name!r}; choose from {sorted(BACKENDS)}")
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if cls is EdgeTTSBackend:
        kwargs.pop("model", None)       # edge has no model choice
    return cls(**kwargs)
