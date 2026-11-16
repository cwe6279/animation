"""
Satellite: a Xiaozhi-protocol device (an ESP32-S3 board) as the character's ears,
voice and face, with the whole talker loop still running on this host.

The board runs the open xiaozhi-esp32 firmware unchanged. At boot it POSTs its
details to our OTA endpoint and gets back the WebSocket address. After its wake
word it opens the socket, says hello, and streams Opus (16 kHz mono, 60 ms) until
we answer. We stream Opus back (24 kHz, so TTS audio needs no resampling) with a
few JSON events: tts start / sentence_start / stop, stt (what it heard) and llm
emotion (the face). Protocol: xiaozhi-esp32/docs/websocket.md.

Two pieces:
  SatelliteAudioEngine  a BaseAudioEngine whose output is the device and whose mic
                        is the device. The timeline clock is simulated, as in
                        NullAudioEngine, so the pipeline and the local face work as
                        usual; the device plays a network hop behind it.
  SatelliteServer       aiohttp HTTP + WebSocket server on its own thread.

Only one device at a time. The server cannot wake the device (the protocol is
device-initiated), so speech while no device is connected is dropped with a note.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .audio_engine import BaseAudioEngine
from .phoneme_scheduler import Emotion, Viseme

try:
    import opuslib
except ImportError:                                    # pragma: no cover
    opuslib = None

DEVICE_MIC_RATE = 16000          # what the firmware sends
DEVICE_OUT_RATE = 24000          # what we send; the hello tells the device
FRAME_MS = 60
OTA_PATH = "/xiaozhi/ota/"
WS_PATH = "/xiaozhi/v1/"
FIRMWARE_PATH = "/xiaozhi/firmware.bin"
PHOTO_PATH = "/xiaozhi/vision/explain"   # the board uploads its camera photos here

# talker emotions -> names the xiaozhi display knows
EMOTION_MAP = {
    Emotion.NEUTRAL: "neutral",
    Emotion.HAPPY: "happy",
    Emotion.ANGRY: "angry",
    Emotion.ANNOYED: "confused",
    Emotion.SAD: "sad",
    Emotion.SURPRISE: "surprised",
}
EMOTION_WIRE = {e: i for i, e in enumerate(EMOTION_MAP)}      # id on the face header

# A face built with the Clara firmware fork takes a 6-byte header on every audio frame:
# magic 0xCF, version 1, emotion id, then one mouth id per 20 ms of the 60 ms frame.
# The board has six mouths (faces/clara): 0 sil, 1 pp, 2 ee, 3 ah, 4 aa, 5 oo.
FACE_MAGIC = 0xCF
VISEME_WIRE = {
    Viseme.SIL: 0, Viseme.PP: 1, Viseme.FF: 1, Viseme.TH: 3, Viseme.DD: 3, Viseme.KK: 3,
    Viseme.CH: 5, Viseme.SS: 2, Viseme.AA: 4, Viseme.EE: 2, Viseme.OO: 5, Viseme.AH: 3,
}


def firmware_app_version(path: str) -> Optional[str]:
    """The version string in an ESP-IDF app image's descriptor (esp_app_desc_t at 0x20)."""
    try:
        with open(path, "rb") as f:
            f.seek(0x20)
            d = f.read(256)
    except OSError as e:
        print(f"[satellite] cannot read firmware {path}: {e}")
        return None
    if d[:4] != b"\x32\x54\xcd\xab":
        print(f"[satellite] {path} is not an ESP-IDF app image (no app descriptor)")
        return None
    return d[16:48].split(b"\0")[0].decode(errors="replace")


def _resample(pcm: bytes, src: int, dst: int) -> bytes:
    if src == dst or not pcm:
        return pcm
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(round(s.size * dst / src))
    x_old = np.linspace(0.0, 1.0, s.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.clip(np.interp(x_new, x_old, s), -32768, 32767).astype(np.int16).tobytes()


class SpeakerShaper:
    """Tailor speech to a small speaker. Bass below what the cone can reproduce only shakes
    the shell, so it is cut; then the level is lifted under a peak limiter, louder without
    the clipping that distorts. Filter and limiter state carry across chunks, so the pieces
    of a sentence join without clicks."""

    def __init__(self, rate: int, hpf_hz: float = 250.0, boost_db: float = 6.0,
                 ceiling: float = 0.9, taps: int = 241, release_ms: float = 120.0):
        self.rate = rate
        self.hpf_hz, self.boost_db = hpf_hz, boost_db
        self.gain = 10 ** (boost_db / 20)
        self.ceiling = ceiling * 32767
        if hpf_hz > 0:                            # windowed-sinc high-pass, linear phase
            n = np.arange(taps) - (taps - 1) / 2
            fc = hpf_hz / rate
            lp = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
            lp /= lp.sum()
            hp = -lp
            hp[(taps - 1) // 2] += 1.0
            self.kernel = hp.astype(np.float32)
        else:
            self.kernel = None
        self._tail = np.zeros(taps - 1, dtype=np.float32)
        self.block = max(1, rate // 500)          # 2 ms limiter blocks
        self._release = float(np.exp(-self.block / (release_ms / 1000 * rate)))
        self._g = 1.0

    @property
    def enabled(self) -> bool:
        return self.kernel is not None or self.boost_db != 0

    def reset(self) -> None:
        self._tail[:] = 0
        self._g = 1.0

    def process(self, pcm: bytes) -> bytes:
        if not pcm or not self.enabled:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if self.kernel is not None:
            buf = np.concatenate([self._tail, x])
            x = np.convolve(buf, self.kernel, mode="valid")
            self._tail = buf[-(self.kernel.size - 1):]
        y = x * self.gain
        n = y.size
        nb = -(-n // self.block)
        padded = np.concatenate([np.abs(y), np.zeros(nb * self.block - n, dtype=np.float32)])
        peaks = padded.reshape(nb, self.block).max(axis=1)
        target = np.minimum(1.0, self.ceiling / np.maximum(peaks, 1.0))
        gains = np.empty(nb)
        g = self._g
        for i, t in enumerate(target):            # fast attack, slow release; 500 blocks a second
            g = t if t < g else t + (g - t) * self._release
            gains[i] = g
        self._g = g
        centers = np.arange(nb) * self.block + self.block / 2
        y *= np.interp(np.arange(n), centers, gains)
        return np.clip(y, -self.ceiling, self.ceiling).astype(np.int16).tobytes()  # ramps overshoot by a hair


class SatelliteAudioEngine(BaseAudioEngine):
    """Speaker and mic are the connected device; the playback clock is simulated."""

    def __init__(self, server: "SatelliteServer", sample_rate: int = DEVICE_OUT_RATE,
                 sync_offset: float = 0.0, lookahead: float = 0.5, clock=time.monotonic,
                 bass_cut_hz: float = 300.0, boost_db: float = 6.0, ceiling: float = 0.5):
        if opuslib is None:
            raise RuntimeError("opuslib is not installed (pip install opuslib; needs libopus)")
        self.server = server
        self.sample_rate = sample_rate
        self.sync_offset = sync_offset
        self.lookahead = lookahead                 # how far ahead of the clock frames are sent
        self._clock = clock
        self._lock = threading.Lock()
        self._t0 = clock()
        self._queue_end = 0                        # frames, timeline
        self._pending: Deque[Tuple[int, bytes]] = deque()   # (start_frame, pcm) in order
        self._carry = b""
        self._carry_start = 0
        self.frame_samples = sample_rate * FRAME_MS // 1000
        self._enc = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_AUDIO)
        self._dec = opuslib.Decoder(DEVICE_MIC_RATE, 1)
        self._sent_rms: Deque[Tuple[float, float]] = deque(maxlen=64)   # (start_t, rms)
        self._on_frames: Optional[Callable[[bytes], None]] = None
        self._mic_rate = DEVICE_MIC_RATE
        self.listening = False
        self.speaking = False                      # we told the device tts start
        self._stop_pending = False
        self._hold_until = 0.0
        self._subtitles: List[Tuple[float, str]] = []
        self._last_emotion: Optional[str] = None
        # wired by voice_loop: the schedule (for emotions), is-the-pipeline-busy, barge-in
        self.schedule = None
        self.busy: Callable[[], bool] = lambda: False
        self.on_abort: Callable[[], None] = lambda: None
        self.on_wake: Callable[[str], None] = lambda text: None
        self.device_face = False                   # the board draws Clara's face: send mouth shapes
        self.idle_close_s = 30.0                   # listening with nothing said -> close, device idles
        self.mic_mute_s = 0.4                      # after the device resumes listening: her own tail
        self._mic_mute_until = 0.0
        self._idle_deadline: Optional[float] = None
        self._end_after_speech = False             # the conversation ended: idle once she has finished
        self._dropped = 0
        self.shaper = SpeakerShaper(sample_rate, hpf_hz=bass_cut_hz, boost_db=boost_db, ceiling=ceiling)
        server.engine = self

    # -- timeline (same model as NullAudioEngine) -------------------------------
    def _frames_out(self) -> int:
        return int((self._clock() - self._t0) * self.sample_rate)

    def open(self, sample_rate: Optional[int] = None) -> None:
        pass                                       # fixed rate: the hello promised it

    def enqueue_pcm(self, pcm: bytes, input_rate: Optional[int] = None) -> float:
        if input_rate and input_rate != self.sample_rate:
            pcm = _resample(pcm, input_rate, self.sample_rate)
        pcm = self.shaper.process(pcm)
        frames = len(pcm) // 2
        with self._lock:
            self._queue_end = max(self._queue_end, self._frames_out())
            start = self._queue_end
            self._queue_end += frames
            if self.server.connected:
                self._pending.append((start, pcm))
            else:
                self._dropped += frames
            return start / self.sample_rate

    def timeline_time(self) -> float:
        return self._frames_out() / self.sample_rate + self.sync_offset

    def queued_seconds(self) -> float:
        with self._lock:
            return max(0.0, (self._queue_end - self._frames_out()) / self.sample_rate)

    def flush(self) -> None:
        with self._lock:
            self._queue_end = self._frames_out()
            self._pending.clear()
            self._carry = b""
            self._subtitles.clear()
            self.shaper.reset()
            if self.speaking:
                self._stop_pending = True

    def get_state(self) -> Tuple[float, float]:
        t = self.timeline_time()
        rms = 0.0
        for start, r in reversed(self._sent_rms):
            if start <= t:
                rms = r if t < start + FRAME_MS / 1000 else 0.0
                break
        return rms, t

    def close(self) -> None:
        pass

    # -- mic from the device ---------------------------------------------------
    def start_mic(self, on_frames=None, rate: Optional[int] = None,
                  device: Optional[int] = None, open_rate: Optional[int] = None) -> None:
        self._on_frames = on_frames
        self._mic_rate = rate or DEVICE_MIC_RATE

    def mic_opus(self, packet: bytes) -> None:
        """A 60 ms Opus frame from the device (server thread)."""
        if self._on_frames is None or not self.listening:
            return
        if self._clock() < self._mic_mute_until:      # the tail of her own voice, still in the mic buffers
            return
        try:
            pcm = self._dec.decode(packet, DEVICE_MIC_RATE * FRAME_MS // 1000)
        except Exception as e:
            print(f"[satellite] opus decode failed: {e}")
            return
        self._on_frames(_resample(pcm, DEVICE_MIC_RATE, self._mic_rate))

    # -- hooks from the pipeline / loop ----------------------------------------
    def note_sentence(self, text: str, t0: float) -> None:
        """Subtitle for the device, shown when its audio starts on the timeline."""
        with self._lock:
            self._subtitles.append((t0, text))

    def note_user_text(self) -> None:
        self._idle_deadline = None

    # -- called by the server's loop every tick ---------------------------------
    def tick(self) -> List[Tuple[str, object]]:
        """Returns a list of ("json", dict) / ("audio", bytes) / ("close", None) to send, in order."""
        out: List[Tuple[str, object]] = []
        now_f = self._frames_out()
        horizon = now_f + int(self.lookahead * self.sample_rate)
        with self._lock:
            if self._stop_pending:
                self._stop_pending = False
                self.speaking = False
                self.server.on_event("satellite", "tts stop (interrupted)")
                out.append(("json", {"type": "tts", "state": "stop"}))
                self._idle_deadline = self._clock() + self.idle_close_s
            # The device drops audio until its state machine has acted on "tts start", so the
            # first frames wait a moment after it (they are still well ahead of playback).
            if self._pending and not self.speaking:
                self.speaking = True
                self._last_emotion = None
                self._idle_deadline = None
                self._hold_until = self._clock() + 0.15
                self.server.on_event("satellite", "tts start")
                out.append(("json", {"type": "tts", "state": "start"}))
                return out
            if self._clock() < self._hold_until:
                return out
            # pull the pcm chunks whose time has come into the carry, in timeline order
            # (a chunk can be a whole sentence; frames leave only as the horizon reaches them,
            # since the device holds about 1.2 s and drops what arrives beyond that)
            frame_bytes = self.frame_samples * 2
            frames: List[Tuple[int, bytes]] = []
            while self._pending and self._pending[0][0] <= horizon:
                start, pcm = self._pending.popleft()
                if not self._carry:
                    self._carry_start = start
                else:
                    expected = self._carry_start + len(self._carry) // 2
                    if start > expected:               # a gap on the timeline: silence keeps alignment
                        self._carry += b"\0\0" * (start - expected)
                self._carry += pcm
            while len(self._carry) >= frame_bytes and self._carry_start <= horizon:
                frames.append((self._carry_start, self._carry[:frame_bytes]))
                self._carry = self._carry[frame_bytes:]
                self._carry_start += self.frame_samples
            # a short tail with nothing queued behind it: pad and send once the pipeline is
            # idle or its moment on the timeline has come, whichever is first
            if self._carry and len(self._carry) < frame_bytes and not self._pending \
                    and (not self.busy() or now_f >= self._carry_start):
                frames.append((self._carry_start, self._carry + b"\0" * (frame_bytes - len(self._carry))))
                self._carry = b""
            subs = []
            while self._subtitles and self._subtitles[0][0] * self.sample_rate <= horizon:
                subs.append(self._subtitles.pop(0))
            drained = not self._pending and not self._carry and now_f >= self._queue_end
        for t0, text in subs:
            out.append(("json", {"type": "tts", "state": "sentence_start", "text": text}))
        for start, pcm in frames:
            t = start / self.sample_rate
            emo_id = 0
            if self.schedule is not None:
                try:
                    cur = self.schedule.current_emotion(t + self.sync_offset)
                    emo, emo_id = EMOTION_MAP.get(cur, "neutral"), EMOTION_WIRE.get(cur, 0)
                except Exception:
                    emo = None
                if emo and emo != self._last_emotion:
                    self._last_emotion = emo
                    out.append(("json", {"type": "llm", "emotion": emo, "text": ""}))
            s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            self._sent_rms.append((t, float(np.sqrt(np.mean(s * s)) / 32768.0) if s.size else 0.0))
            try:
                packet = self._enc.encode(pcm, self.frame_samples)
            except Exception as e:
                print(f"[satellite] opus encode failed: {e}")
                continue
            if self.device_face:
                packet = self._face_header(t, emo_id) + packet
            out.append(("audio", packet))
        if self.speaking and drained and not self.busy():
            self.speaking = False
            self.server.on_event("satellite", f"tts stop (played {self._queue_end / self.sample_rate:.1f} s on the timeline)")
            out.append(("json", {"type": "tts", "state": "stop"}))
            self._idle_deadline = self._clock() + (0.0 if self._end_after_speech else self.idle_close_s)
        if self._idle_deadline is not None and self._clock() > self._idle_deadline and not self.busy():
            self._idle_deadline = None
            self._end_after_speech = False
            out.append(("close", None))
        if self._dropped:
            print(f"[satellite] no device connected: dropped {self._dropped / self.sample_rate:.1f} s of speech")
            self._dropped = 0
        return out

    def end_conversation(self) -> None:
        """The brain ended the conversation (goodbye, "stop listening", "go to sleep"): close the
        channel as soon as her last words have played, so the board goes back to waiting for its
        wake word instead of listening on for the usual idle timeout."""
        with self._lock:
            self._end_after_speech = True
            if not self.speaking and not self._pending:
                self._idle_deadline = self._clock()

    def _face_header(self, t: float, emo_id: int) -> bytes:
        """Mouth shape for each 20 ms of the frame starting at timeline time t."""
        ids = [0, 0, 0]
        if self.schedule is not None:
            for i in range(3):
                try:
                    ids[i] = VISEME_WIRE.get(self.schedule.current_viseme(t + self.sync_offset + 0.01 + i * 0.02), 0)
                except Exception:
                    break
        return bytes([FACE_MAGIC, 1, emo_id & 0x7f, ids[0], ids[1], ids[2]])

    # -- events from the device (server thread) ---------------------------------
    def device_listen(self, state: str, mode: str, text: str) -> None:
        if state == "start":
            self.listening = True
            self._mic_mute_until = self._clock() + self.mic_mute_s
            self._idle_deadline = self._clock() + self.idle_close_s
        elif state == "stop":
            self.listening = False
        elif state == "detect":
            if self.speaking or self.busy():
                self.on_abort()
            self.on_wake(text)

    def device_abort(self, reason: str) -> None:
        self.on_abort()

    def device_event(self) -> None:
        """The board opened the channel to report an event: if nothing comes of it, close as
        after a wake word with nothing said."""
        if self._idle_deadline is None:
            self._idle_deadline = self._clock() + self.idle_close_s

    def device_closed(self) -> None:
        self.listening = False
        self.speaking = False
        self._idle_deadline = None
        self._end_after_speech = False
        with self._lock:
            self._pending.clear()
            self._carry = b""
            self._subtitles.clear()


def with_port(host: str, port: int) -> str:
    """host:port for an address the board can reach. The board's websocket request names the host
    without a port, and an upload URL without one goes to port 80."""
    host = (host or "").strip()
    if not host:
        return ""
    if host.startswith("["):                       # [ipv6] or [ipv6]:port
        return host if "]:" in host else f"{host}:{port}"
    return host if ":" in host else f"{host}:{port}"


class SatelliteServer:
    """HTTP (OTA config) + WebSocket (audio + events) for one xiaozhi device."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8030, advertise_host: Optional[str] = None,
                 token: str = "clara", on_event: Optional[Callable[[str, str], None]] = None,
                 firmware: Optional[str] = None, volume: Optional[int] = None):
        self.host, self.port = host, port
        self.volume = volume                     # device speaker level to set on connect, 0-100
        self.advertise_host = advertise_host     # None: whatever address the device reached us on
        self.token = token
        # A firmware image (build/xiaozhi.bin) to offer over the air: the device upgrades
        # when the image's version is newer than its own.
        self.firmware = firmware
        self.firmware_version = firmware_app_version(firmware) if firmware else None
        self.on_event = on_event or (lambda kind, text: print(f"[satellite] {kind}: {text}"))
        self.engine: Optional[SatelliteAudioEngine] = None
        self.device: Dict[str, str] = {}
        self.device_status: Dict[str, object] = {}  # battery, network, chip, from the board's status tool
        self.on_device_status: Optional[Callable[[str], None]] = None   # one line, each time it connects
        # the board was picked up, shaken, tilted: (kind, phrase), already sampled by the board
        self.on_board_event: Optional[Callable[[str, str], None]] = None
        # The board's camera (board_eyes.BoardEyes wires these): describe(jpeg, question) -> text,
        # and a call when the board says it could not take the photo.
        self.photo_handler: Optional[Callable[[bytes, str], str]] = None
        self.photo_failed: Optional[Callable[[str], None]] = None
        self.device_camera = False               # the board listed a camera tool
        self._ws_host = ""                        # host:port the board reached us on
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.session_id = ""

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="satellite", daemon=True)
        self._thread.start()

    def send_json(self, msg: dict) -> None:
        """From any thread."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send_json(msg)))

    def look(self, where: str, hold_ms: int = 2500) -> None:
        """Point her gaze: left, right, up, down or ahead (the viewer's left and right)."""
        self.send_json({"type": "face", "look": where, "hold_ms": hold_ms})

    def send_stt(self, text: str) -> None:
        self.send_json({"type": "stt", "text": text})

    def take_photo(self, question: str = "") -> bool:
        """Ask the board's camera for a photo (from any thread). The photo arrives at PHOTO_PATH
        and goes to photo_handler. False when no board is connected."""
        if not self.connected:
            return False
        self.send_json({"type": "mcp", "session_id": self.session_id, "payload": {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "self.camera.take_photo", "arguments": {"question": question or "What do you see?"}}}})
        return True

    # -- the loop ----------------------------------------------------------------
    def _run(self) -> None:
        from aiohttp import web
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_post(OTA_PATH, self._ota)
        app.router.add_get(OTA_PATH, self._ota)
        app.router.add_get(WS_PATH, self._websocket)
        app.router.add_get(FIRMWARE_PATH, self._firmware)
        app.router.add_post(PHOTO_PATH, self._photo)
        app.router.add_get("/", self._index)
        runner = web.AppRunner(app, access_log=None)
        self._loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, self._host_for_bind(), self.port)
        self._loop.run_until_complete(site.start())
        print(f"[satellite] listening on {self.host}:{self.port}  "
              f"device setup -> OTA URL http://<this host>:{self.port}{OTA_PATH}")
        self._loop.create_task(self._ticker())
        self._loop.run_forever()

    def _host_for_bind(self) -> str:
        return self.host

    async def _index(self, request):
        from aiohttp import web
        return web.json_response({"service": "talker satellite", "device": self.device,
                                  "status": self.device_status, "connected": self.connected,
                                  "ota": OTA_PATH, "websocket": WS_PATH})

    async def _ota(self, request):
        from aiohttp import web
        info = {}
        if request.method == "POST":
            try:
                info = await request.json()
            except Exception:
                info = {}
        app_info = info.get("application", {}) if isinstance(info, dict) else {}
        board = info.get("board", {}) if isinstance(info, dict) else {}
        self.device = {
            "mac": request.headers.get("Device-Id", ""),
            "client": request.headers.get("Client-Id", ""),
            "version": app_info.get("version", ""),
            "board": board.get("type", info.get("board_type", "")) if isinstance(board, dict) else "",
            "ip": board.get("ip", "") if isinstance(board, dict) else "",
            "seen": time.strftime("%H:%M:%S"),
        }
        host = self.advertise_host or request.host.split(":")[0]
        ws_url = f"ws://{host}:{self.port}{WS_PATH}"
        self.on_event("satellite", f"device {self.device['mac'] or '?'} v{self.device['version'] or '?'} "
                                   f"checked in; websocket {ws_url}")
        now = time.time()
        offset_min = -time.timezone // 60 if not time.daylight else -time.altzone // 60
        if self.firmware_version:
            # offered over the air; the device takes it only if the version is newer than its own
            firmware = {"version": self.firmware_version, "url": f"http://{host}:{self.port}{FIRMWARE_PATH}"}
        else:
            # same version as it has: "current is the latest", no upgrade
            firmware = {"version": self.device["version"] or "0.0.0", "url": ""}
        return web.json_response({
            "server_time": {"timestamp": int(now * 1000), "timezone_offset": offset_min},
            "firmware": firmware,
            "websocket": {"url": ws_url, "token": self.token, "version": 1},
        })

    async def _firmware(self, request):
        from aiohttp import web
        if not self.firmware:
            raise web.HTTPNotFound()
        self.on_event("satellite", f"device is downloading firmware {self.firmware_version}")
        return web.FileResponse(self.firmware, headers={"Content-Type": "application/octet-stream"})

    async def _websocket(self, request):
        from aiohttp import web
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=1 << 20)
        await ws.prepare(request)
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = ws
        self._ws_host = with_port(request.host, self.port)
        self.session_id = uuid.uuid4().hex[:8]
        eng = self.engine
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    if eng is not None:
                        eng.mic_opus(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except ValueError:
                        continue
                    await self._on_json(ws, data)
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    break
        finally:
            if self._ws is ws:
                self._ws = None
            if eng is not None:
                eng.device_closed()
            self.on_event("satellite", "device disconnected")
        return ws

    async def _on_json(self, ws, data: dict) -> None:
        kind = data.get("type")
        eng = self.engine
        if kind == "hello":
            ap = data.get("audio_params", {})
            face = bool((data.get("features") or {}).get("clara_face"))
            if eng is not None:
                eng.device_face = face
            self.on_event("satellite", f"device connected (mic {ap.get('sample_rate')} Hz, "
                                       f"{ap.get('frame_duration')} ms frames"
                                       f"{', draws her face' if face else ''})")
            await ws.send_json({
                "type": "hello", "transport": "websocket", "session_id": self.session_id,
                "audio_params": {"format": "opus", "sample_rate": DEVICE_OUT_RATE,
                                 "channels": 1, "frame_duration": FRAME_MS},
            })
            if self.volume is not None:            # the device's speaker, 0-100, through its MCP tool
                await ws.send_json({"type": "mcp", "session_id": self.session_id, "payload": {
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "self.audio_speaker.set_volume", "arguments": {"volume": int(self.volume)}}}})
            if self.photo_handler is not None:
                # MCP handshake: tells the board where to upload camera photos (and asks for its tools)
                host = f"{self.advertise_host}:{self.port}" if self.advertise_host else self._ws_host
                await ws.send_json({"type": "mcp", "session_id": self.session_id, "payload": {
                    "jsonrpc": "2.0", "id": 4, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {"vision": {"url": f"http://{host}{PHOTO_PATH}", "token": self.token}}}}})
            # battery, signal, temperature: the answer arrives as an "mcp" message (id 2)
            await ws.send_json({"type": "mcp", "session_id": self.session_id, "payload": {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "self.get_device_status", "arguments": {}}}})
        elif kind == "listen":
            state, mode, text = data.get("state", ""), data.get("mode", ""), data.get("text", "")
            if state == "detect":
                self.on_event("satellite", f"wake word: {text}")
            else:
                busy = eng is not None and (eng.speaking or eng.busy())
                self.on_event("satellite", f"device listen {state} ({mode})"
                                           + (" while she is still speaking" if busy else ""))
            if eng is not None:
                eng.device_listen(state, mode, text)
        elif kind == "abort":
            self.on_event("satellite", f"device abort: {data.get('reason', '')}")
            if eng is not None:
                eng.device_abort(data.get("reason", ""))
        elif kind == "mcp":
            self._on_mcp(data.get("payload") or {})
        elif kind == "event":
            k, text = str(data.get("kind", "")), str(data.get("text") or data.get("kind", ""))
            self.on_event("satellite", f"board event: {text}")
            if eng is not None:
                eng.device_event()
            if self.on_board_event is not None:
                self.on_board_event(k, text)
        else:
            self.on_event("satellite", f"device sent {kind}: {json.dumps(data)[:120]}")

    def _on_mcp(self, payload: dict) -> None:
        """A reply to one of our tool calls (id 2 = device status, 3 = a photo, 4 = the handshake)."""
        if payload.get("id") == 3:
            res = payload.get("result") or {}
            err = payload.get("error") or (res if isinstance(res, dict) and res.get("isError") else None)
            if err and self.photo_failed is not None:
                self.photo_failed(json.dumps(err)[:200])
            return
        if payload.get("id") != 2:
            return
        result = payload.get("result")
        if isinstance(result, dict) and "content" in result:          # MCP: content[0].text holds JSON
            try:
                result = json.loads(result["content"][0]["text"])
            except (KeyError, IndexError, TypeError, ValueError):
                result = None
        if not isinstance(result, dict):
            return
        self.device_status = result
        line = self.status_line()
        self.on_event("satellite", line)
        if self.on_device_status is not None:
            try:
                self.on_device_status(line)
            except Exception as e:
                print(f"[satellite] on_device_status failed: {e}")

    async def _photo(self, request):
        """The board's camera upload: multipart 'question' + 'file' (JPEG). The answer text becomes
        the take_photo tool's result on the board."""
        from aiohttp import web
        question, jpeg = "", b""
        try:
            reader = await request.multipart()
            async for part in reader:
                if part.name == "question":
                    question = (await part.read(decode=True)).decode(errors="replace")
                elif part.name == "file":
                    jpeg = bytes(await part.read(decode=False))
        except Exception as e:
            self.on_event("satellite", f"photo upload failed: {e}")
            return web.json_response({"success": False, "message": "bad upload"}, status=400)
        self.on_event("satellite", f"photo from the board: {len(jpeg) // 1024} KB")
        if not jpeg or self.photo_handler is None:
            return web.json_response({"success": False, "message": "no photo handler"}, status=503)
        text = await asyncio.get_running_loop().run_in_executor(None, self.photo_handler, jpeg, question)
        return web.json_response({"success": True, "text": text})

    def status_line(self) -> str:
        """One sentence about the board, for the log and for her context."""
        st = self.device_status or {}
        parts = []
        b = st.get("battery") or {}
        if "level" in b:
            volts = f" ({b['voltage']:.2f} V)" if isinstance(b.get("voltage"), (int, float)) else ""
            parts.append(f"battery {b['level']}%{volts}" + (", charging" if b.get("charging") else ", on battery"))
        n = st.get("network") or {}
        if n.get("signal"):
            parts.append(f"wifi signal {n['signal']}")
        c = st.get("chip") or {}
        if "temperature" in c:
            parts.append(f"chip {c['temperature']:.0f} C")
        return "the board reports " + ", ".join(parts) if parts else "the board reported no status"

    async def _send_json(self, msg: dict) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return
        msg = dict(msg)
        msg.setdefault("session_id", self.session_id)
        try:
            await ws.send_json(msg)
        except Exception as e:
            print(f"[satellite] send failed: {e}")

    async def _ticker(self) -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            # A tick that comes late means the host stalled (paging, the GIL held by something
            # heavy): frames reach the board late and, past its lead, a gap is heard.
            if now - last > 0.25 and self.engine is not None and self.engine.speaking:
                self.on_event("satellite", f"host stalled {now - last:.2f} s while she was speaking")
            last = now
            eng, ws = self.engine, self._ws
            if eng is None or ws is None or ws.closed:
                continue
            try:
                for what, payload in eng.tick():
                    if what == "json":
                        await self._send_json(payload)
                    elif what == "audio":
                        await ws.send_bytes(payload)
                    elif what == "close":
                        self.on_event("satellite", "quiet: closing the channel, device idles")
                        await ws.close()
                        break
            except Exception as e:
                print(f"[satellite] tick failed: {e}")
