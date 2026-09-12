"""
talker/audio_engine.py
===============
One persistent output stream with a frame-accurate timeline.

The old design opened a PyAudio stream per utterance and timed lip sync with
wall-clock-since-first-write. Here the stream is opened once and driven by a
PortAudio callback that pulls from a byte queue; when the queue is empty it
emits silence, so the stream never stops and the clock never resets.

Timeline:
    frames_out   frames handed to the device so far (audio + silence)
    queue_end    frame index where the next enqueued chunk will start
    enqueue_pcm() returns the timeline time (seconds) its chunk will start
    timeline_time() returns the time currently audible (frames_out, plus time
                    since the last callback, minus device latency, plus the
                    user's sync_offset)

The speech pipeline uses enqueue_pcm()'s return value to place viseme events;
the renderer reads timeline_time() every frame.

NullAudioEngine has the same interface with no device (tests, --no-audio).
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
import wave
from typing import Optional, Tuple

import numpy as np

try:
    import pyaudio
except ImportError:      # let the rest of the app import without a sound device
    pyaudio = None


def _resample_int16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if s.size == 0 or src_rate == dst_rate:
        return pcm
    n_out = int(round(s.size * dst_rate / src_rate))
    out = np.interp(np.linspace(0.0, 1.0, n_out, endpoint=False),
                    np.linspace(0.0, 1.0, s.size, endpoint=False), s)
    return out.astype(np.int16).tobytes()


def _quiet_pyaudio():
    """Construct PyAudio without ALSA's harmless 'Unknown PCM ...' chatter on stderr."""
    import contextlib
    try:
        stderr_fd = sys.stderr.fileno()
        saved = os.dup(stderr_fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                return pyaudio.PyAudio()
            finally:
                os.dup2(saved, stderr_fd)
                os.close(saved)
    except Exception:
        return pyaudio.PyAudio()


def _rms_int16(data: bytes) -> float:
    if not data:
        return 0.0
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


class BaseAudioEngine:
    sample_rate: int = 24000

    # Microphone filtering, in Hz; 0 disables either stage. See talker/audio_filters.py.
    # The high-pass takes out room rumble before anything measures the level; the
    # low-pass is the anti-aliasing stage used when the device runs faster than the
    # recognizer wants. Both can be changed while the loop is running.
    mic_high_pass: float = 90.0
    mic_low_pass: float = 7500.0

    # -- timeline ------------------------------------------------------
    def enqueue_pcm(self, pcm: bytes, input_rate: Optional[int] = None) -> float: ...
    def timeline_time(self) -> float: ...
    def queued_seconds(self) -> float: ...
    def flush(self) -> None: ...
    def get_state(self) -> Tuple[float, float]:
        """(rms, timeline_time) for the render loop."""
        return 0.0, self.timeline_time()

    def output_envelope(self) -> list:
        return []

    # -- lifecycle -----------------------------------------------------
    def open(self, sample_rate: Optional[int] = None) -> None: ...
    def close(self) -> None: ...
    def list_input_devices(self) -> list:
        return []

    def list_output_devices(self) -> list:
        return []

    def start_mic(self, on_frames=None, rate: Optional[int] = None,
                  device: Optional[int] = None, open_rate: Optional[int] = None) -> None:
        """Open an input device. on_frames(pcm_int16_bytes) is called per chunk."""
        raise RuntimeError("pyaudio is not installed, so no audio device can be opened "
                           "(dnf install python3-devel portaudio-devel; pip install pyaudio)")

    # -- convenience ---------------------------------------------------
    def play_wav(self, path: str, stop_event: Optional[threading.Event] = None) -> Tuple[float, float]:
        """
        Stream a WAV file onto the timeline (any width/channels; resampled to
        the engine rate with linear interpolation). Blocks while feeding with
        ~1.5 s of lookahead so an interrupt can flush promptly.
        Returns (start_time, end_time) on the timeline.
        """
        with wave.open(path, "rb") as wf:
            rate, nch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            start = None
            end = 0.0
            while True:
                raw = wf.readframes(4096)
                if not raw:
                    break
                pcm = self._to_engine_pcm(raw, rate, nch, sw)
                if not pcm:
                    continue
                while self.queued_seconds() > 1.5:
                    if stop_event is not None and stop_event.is_set():
                        return (start or self.timeline_time(), end)
                    time.sleep(0.02)
                t = self.enqueue_pcm(pcm)
                if start is None:
                    start = t
                end = t + len(pcm) / 2 / self.sample_rate
        return (start if start is not None else self.timeline_time(), end)

    def _to_engine_pcm(self, raw: bytes, rate: int, nch: int, sw: int) -> bytes:
        if sw == 1:
            s = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
        elif sw == 2:
            s = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        elif sw == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            s = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 256.0
        elif sw == 4:
            s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 65536.0
        else:
            raise ValueError(f"Unsupported sample width {sw}")
        if nch > 1:
            s = s.reshape(-1, nch).mean(axis=1)
        if rate != self.sample_rate and s.size:
            n_out = int(round(s.size * self.sample_rate / rate))
            x_old = np.linspace(0.0, 1.0, s.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            s = np.interp(x_new, x_old, s)
        return np.clip(s, -32768, 32767).astype(np.int16).tobytes()


# ─────────────────────────────────────────────────────
# REAL DEVICE
# ─────────────────────────────────────────────────────
class AudioEngine(BaseAudioEngine):
    FRAMES_PER_BUFFER = 480     # 20 ms at 24 kHz

    def __init__(self, sample_rate: int = 24000, sync_offset: float = 0.0,
                 output_device=None):
        if pyaudio is None:
            raise RuntimeError("pyaudio is not installed (pip install pyaudio)")
        self.sample_rate = sample_rate
        self.sync_offset = sync_offset       # seconds; + delays the face, - advances it
        self._pa = _quiet_pyaudio()
        # Device index, or a case-insensitive name fragment ("jabra"); None = default.
        self.output_device = self.resolve_device(output_device, "output")
        self._lock = threading.Lock()
        self._buf: "collections.deque[bytes]" = collections.deque()
        self._buf_bytes = 0
        self._frames_out = 0
        self._queue_end = 0
        self._last_cb = time.monotonic()
        self._rms = 0.0
        self._latency = 0.0
        self._stream = None
        self._mic_stream = None
        self._mic_rms = 0.0
        # (monotonic time, output RMS) per callback, last ~2 s: the playback envelope
        self._out_env: "collections.deque[tuple]" = collections.deque(maxlen=150)

    # -- output stream -------------------------------------------------
    def open(self, sample_rate: Optional[int] = None) -> None:
        """
        Open the output stream, preferably at `sample_rate` (the rate the TTS
        audio arrives at). Devices that refuse it (e.g. a USB speakerphone
        fixed at 48 kHz) are opened at a rate they accept and everything
        enqueued is resampled to it; self.sample_rate is the opened rate.
        """
        want = sample_rate or self.sample_rate
        if self._stream is not None:
            if want == self.sample_rate:
                return
            self.close_output()
        try:
            info = (self._pa.get_device_info_by_index(self.output_device) if self.output_device is not None
                    else self._pa.get_default_output_device_info())
            native = int(info.get("defaultSampleRate", 48000))
            name = info.get("name", "default")
        except Exception:
            native, name = 48000, "default"
        last_err = None
        candidates = [self.output_device, None] + [i for i, n, _, _ in self.list_output_devices()
                                                   if "hw:" not in n]
        for dev in dict.fromkeys(candidates):
            for rate in dict.fromkeys([want, native, 48000, 44100, 24000, 16000]):
                try:
                    self._stream = self._pa.open(
                        format=pyaudio.paInt16, channels=1, rate=rate, output=True,
                        output_device_index=dev,
                        frames_per_buffer=int(rate * 0.02), stream_callback=self._callback)
                    break
                except Exception as e:
                    last_err = e
                    self._stream = None
            if self._stream is not None:
                if dev != self.output_device:
                    print("[audio] requested output device unavailable; using another")
                try:
                    name = (self._pa.get_device_info_by_index(dev) if dev is not None
                            else self._pa.get_default_output_device_info()).get("name", "default")
                except Exception:
                    pass
                break
        if self._stream is None:
            raise RuntimeError(f"could not open any output device: {last_err}")
        with self._lock:
            # Rebase the timeline so times stay in seconds across a rate change.
            secs_out = self._frames_out / self.sample_rate
            self.sample_rate = rate
            self._frames_out = int(secs_out * rate)
            self._queue_end = self._frames_out
            self._buf.clear()
            self._buf_bytes = 0
            self._last_cb = time.monotonic()
        self._latency = float(self._stream.get_output_latency() or 0.0)
        self._stream.start_stream()
        note = "" if rate == want else f" (device refused {want} Hz; resampling)"
        print(f"[audio] output open: {name} @ {rate} Hz, device latency {self._latency*1000:.0f} ms{note}")

    def _callback(self, in_data, frame_count, time_info, status):
        need = frame_count * 2
        out = bytearray()
        with self._lock:
            while need > 0 and self._buf:
                chunk = self._buf[0]
                if len(chunk) <= need:
                    out += chunk
                    need -= len(chunk)
                    self._buf.popleft()
                else:
                    out += chunk[:need]
                    self._buf[0] = chunk[need:]
                    need = 0
            self._buf_bytes -= len(out)
            if need > 0:
                out += bytes(need)          # underrun / idle: silence
            self._frames_out += frame_count
            if not self._buf:
                self._queue_end = self._frames_out
            self._last_cb = time.monotonic()
            self._rms = _rms_int16(bytes(out)) if len(out) > need else 0.0
            self._out_env.append((self._last_cb, self._rms))
        return (bytes(out), pyaudio.paContinue)

    def enqueue_pcm(self, pcm: bytes, input_rate: Optional[int] = None) -> float:
        if self._stream is None:
            self.open(input_rate)
        if input_rate and input_rate != self.sample_rate:
            pcm = _resample_int16(pcm, input_rate, self.sample_rate)
        with self._lock:
            start = self._queue_end / self.sample_rate
            self._buf.append(pcm)
            self._buf_bytes += len(pcm)
            self._queue_end += len(pcm) // 2
            return start

    def timeline_time(self) -> float:
        with self._lock:
            elapsed = min(time.monotonic() - self._last_cb, 0.02)
            t = self._frames_out / self.sample_rate + elapsed - self._latency + self.sync_offset
        return max(0.0, t)

    def queued_seconds(self) -> float:
        with self._lock:
            return self._buf_bytes / 2 / self.sample_rate

    def flush(self) -> None:
        with self._lock:
            self._buf.clear()
            self._buf_bytes = 0
            self._queue_end = self._frames_out

    def get_state(self) -> Tuple[float, float]:
        with self._lock:
            rms = max(self._rms, self._mic_rms)
        return rms, self.timeline_time()

    def output_envelope(self) -> list:
        """[(time, rms), ...] of what the speaker played recently (for echo detection)."""
        with self._lock:
            return list(self._out_env)

    # -- mic input (amplitude mode) -----------------------------------
    def resolve_device(self, spec, kind: str) -> Optional[int]:
        """
        spec: None (default), an int index, a digit string, or a name fragment.
        Indices shift whenever a device is plugged in, so names are safer.
        """
        if spec is None or spec == "":
            return None
        if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
            return int(spec)
        devices = self.list_input_devices() if kind == "input" else self.list_output_devices()
        needle = str(spec).lower()
        matches = [(i, n) for i, n, _, _ in devices if needle in n.lower()]
        if not matches:
            names = ", ".join(n for _, n, _, _ in devices)
            raise RuntimeError(f"no {kind} device matching {spec!r}; available: {names}")
        # Prefer the audio-server (PipeWire/Pulse) entry over raw hw: entries
        matches.sort(key=lambda m: ("hw:" in m[1], m[0]))
        return matches[0][0]

    def resolve_devices(self, spec, kind: str) -> list:
        """Every device whose name matches, best first. A fragment like "gomic" can match
        both an audio-server node and the raw hardware behind it, and either one may turn
        out to be unusable, so the caller tries them in turn."""
        if spec is None or spec == "":
            return [None]
        if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
            return [int(spec)]
        devices = self.list_input_devices() if kind == "input" else self.list_output_devices()
        needle = str(spec).lower()
        matches = [(i, n) for i, n, _, _ in devices if needle in n.lower()]
        if not matches:
            names = ", ".join(n for _, n, _, _ in devices)
            raise RuntimeError(f"no {kind} device matching {spec!r}; available: {names}")
        matches.sort(key=lambda m: ("hw:" in m[1], m[0]))
        return [i for i, _ in matches]

    OPEN_TIMEOUT_S = 4.0

    def _open_stream(self, **kw):
        """PortAudio can block forever opening some devices rather than returning an
        error: a PipeWire node whose rate the server will not accept does exactly that,
        and it hangs the whole start-up. Open on a worker thread and give up on it."""
        box: dict = {}

        def work():
            try:
                box["stream"] = self._pa.open(**kw)
            except Exception as e:                      # pragma: no cover - driver dependent
                box["error"] = e
        t = threading.Thread(target=work, name="pa-open", daemon=True)
        t.start()
        t.join(self.OPEN_TIMEOUT_S)
        if t.is_alive():
            raise TimeoutError(f"the driver did not respond within {self.OPEN_TIMEOUT_S:.0f} s")
        if "error" in box:
            raise box["error"]
        return box["stream"]

    def list_input_devices(self) -> list:
        """[(index, name, default_rate, is_default), ...] for devices with input channels."""
        return self._list_devices("maxInputChannels", "get_default_input_device_info")

    def list_output_devices(self) -> list:
        return self._list_devices("maxOutputChannels", "get_default_output_device_info")

    def _list_devices(self, chan_key: str, default_getter: str) -> list:
        out = []
        try:
            default_idx = getattr(self._pa, default_getter)().get("index")
        except Exception:
            default_idx = None
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if int(info.get(chan_key, 0)) > 0:
                out.append((i, info.get("name", "?"), int(info.get("defaultSampleRate", 0)), i == default_idx))
        return out

    def start_mic(self, on_frames=None, rate: Optional[int] = None,
                  device=None, open_rate: Optional[int] = None) -> None:
        """
        Open an input device (default, or `device`: index or name fragment).
        Amplitude (RMS) is tracked for the amplitude-driven mouth; on_frames(pcm)
        additionally receives raw int16 mono chunks at `rate` (for speech-to-text).
        The device is opened at `open_rate` if given, else at `rate`, else at its
        native rate; whatever rate it opens at is resampled to `rate`.
        """
        asked_for = device                    # keep the name the caller gave, for messages
        wanted = self.resolve_devices(device, "input")
        device = wanted[0]
        try:
            info = (self._pa.get_device_info_by_index(device) if device is not None
                    else self._pa.get_default_input_device_info())
            native = int(info.get("defaultSampleRate", 16000))
            dev_name = info.get("name", "default")
        except Exception as e:
            raise RuntimeError(f"input device not found ({e})")
        want = rate or native

        errors: list = []

        def make_cb(open_rate):
            # Two filter stages, each where it is cheapest and where it belongs.
            # The low-pass has to run before the decimation below, or everything
            # above half the target rate folds back into the speech band. The
            # high-pass runs after, at the lower rate, and takes out the room
            # rumble that would otherwise move the speech gate. Both are retuned
            # live from the engine's attributes, which the control page writes.
            from .audio_filters import MicFilter
            pre = MicFilter(open_rate, low_pass=self.mic_low_pass) if open_rate != want else None
            post = MicFilter(want, high_pass=self.mic_high_pass)

            def cb(in_data, frame_count, time_info, status):
                data = in_data
                if pre is not None:
                    pre.configure(low_pass=self.mic_low_pass)
                    data = pre.process_bytes(data)
                if open_rate != want:
                    s = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    n_out = int(round(s.size * want / open_rate))
                    data = np.interp(np.linspace(0.0, 1.0, n_out, endpoint=False),
                                     np.linspace(0.0, 1.0, s.size, endpoint=False), s
                                     ).astype(np.int16).tobytes()
                post.configure(high_pass=self.mic_high_pass)
                data = post.process_bytes(data)
                # the level the gate and the meters read is the filtered one, so it
                # reflects what the recognizer actually hears
                with self._lock:
                    self._mic_rms = _rms_int16(data)
                if on_frames is not None:
                    try:
                        on_frames(data)
                    except Exception as e:      # never let a consumer kill the stream
                        if not errors:
                            print(f"[audio] mic consumer error (further ones suppressed): {e!r}")
                        errors.append(e)
                return (None, pyaudio.paContinue)
            return cb

        last_err = None
        if wanted[0] is not None:             # last resort: the system default input
            wanted = list(wanted) + [None]
        for dev in wanted:                    # a name can match several; try each in turn
            try:
                dev_name = (self._pa.get_device_info_by_index(dev)["name"] if dev is not None
                            else self._pa.get_default_input_device_info()["name"])
            except Exception as e:
                last_err = e
                continue
            candidates = [open_rate] if open_rate else [want, native, 48000, 44100, 16000]
            for rate_try in dict.fromkeys(candidates):
                try:
                    self._mic_stream = self._open_stream(
                        format=pyaudio.paInt16, channels=1, rate=rate_try, input=True,
                        input_device_index=dev,
                        frames_per_buffer=int(rate_try * 0.064), stream_callback=make_cb(rate_try))
                    self._mic_stream.start_stream()
                    note = "" if rate_try == want else f", resampled to {want} Hz"
                    fallback = ("" if dev is wanted[0] else
                                f"  — NOT {asked_for!r}, which would not open")
                    print(f"[audio] mic open: {dev_name} @ {rate_try} Hz{note}{fallback}")
                    return
                except TimeoutError as e:
                    print(f"[audio] {dev_name} @ {rate_try} Hz: {e}; trying another")
                    last_err = e
                    break                      # this device is wedged; move to the next one
                except Exception as e:
                    last_err = e
        names = ", ".join(n for _, n, _, _ in self.list_input_devices())
        raise RuntimeError(f"could not open input device {dev_name!r}: {last_err}. "
                           f"Available inputs: {names}")

    # -- teardown --------------------------------------------------------
    def close_output(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def close(self) -> None:
        self.close_output()
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
        try:
            self._pa.terminate()
        except Exception:
            pass



# ─────────────────────────────────────────────────────
# ECHO GUARD  (is the mic just hearing the speaker?)
# ─────────────────────────────────────────────────────
class EchoGuard:
    """
    Compares the mic loudness envelope with the speaker's output envelope over
    the last second or so. Echo rises and falls with the playback (after a
    small delay); a person talking over the character does not. Returns the
    best normalised correlation over lags 0-300 ms; above ~0.5 treat the mic
    as echo.
    """

    def __init__(self, window_s: float = 1.2, max_lag_s: float = 0.3, step_s: float = 0.02):
        self.window_s = window_s
        self.max_lag_s = max_lag_s
        self.step_s = step_s
        self._mic: "collections.deque[tuple]" = collections.deque(maxlen=int(window_s / 0.01) + 50)

    def add_mic(self, t: float, rms: float) -> None:
        self._mic.append((t, rms))

    @staticmethod
    def _resample(points, t0, t1, step):
        if not points:
            return None
        ts = np.array([p[0] for p in points]); vs = np.array([p[1] for p in points], dtype=np.float32)
        grid = np.arange(t0, t1, step)
        if grid.size < 8:
            return None
        return np.interp(grid, ts, vs, left=0.0, right=0.0)

    def correlation(self, out_env, now: float) -> float:
        if len(self._mic) < 8 or len(out_env) < 8:
            return 0.0
        t0, t1 = now - self.window_s, now
        mic = self._resample(list(self._mic), t0, t1, self.step_s)
        if mic is None or mic.std() < 1e-3:
            return 0.0
        best = 0.0
        lags = int(self.max_lag_s / self.step_s)
        for lag in range(0, lags + 1):
            out = self._resample(out_env, t0 - lag * self.step_s, t1 - lag * self.step_s, self.step_s)
            if out is None or out.std() < 1e-3:
                continue
            n = min(len(mic), len(out))
            a = mic[:n] - mic[:n].mean(); b = out[:n] - out[:n].mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            if denom > 0:
                best = max(best, float((a * b).sum() / denom))
        return best


# ─────────────────────────────────────────────────────
# NO DEVICE  (tests, headless, --no-audio)
# ─────────────────────────────────────────────────────
class NullAudioEngine(BaseAudioEngine):
    """Simulates playback with the monotonic clock; drops the audio bytes."""

    def __init__(self, sample_rate: int = 24000, sync_offset: float = 0.0, clock=time.monotonic):
        self.sample_rate = sample_rate
        self.sync_offset = sync_offset
        self._clock = clock
        self._lock = threading.Lock()
        self._t0 = clock()
        self._queue_end = 0

    def _frames_out(self) -> int:
        return int((self._clock() - self._t0) * self.sample_rate)

    def open(self, sample_rate: Optional[int] = None) -> None:
        if sample_rate:
            self.sample_rate = sample_rate

    def enqueue_pcm(self, pcm: bytes, input_rate: Optional[int] = None) -> float:
        frames = len(pcm) // 2
        if input_rate and input_rate != self.sample_rate:
            frames = int(round(frames * self.sample_rate / input_rate))
        with self._lock:
            self._queue_end = max(self._queue_end, self._frames_out())
            start = self._queue_end / self.sample_rate
            self._queue_end += frames
            return start

    def timeline_time(self) -> float:
        return self._frames_out() / self.sample_rate + self.sync_offset

    def queued_seconds(self) -> float:
        with self._lock:
            return max(0.0, (self._queue_end - self._frames_out()) / self.sample_rate)

    def flush(self) -> None:
        with self._lock:
            self._queue_end = self._frames_out()

    def close(self) -> None:
        pass
