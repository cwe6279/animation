"""
voice_loop.py — the full round trip: mic -> speech-to-text -> Claude -> voice + face.

    python voice_loop.py --face eve                      # talk to it
    python voice_loop.py --face eve --tts elevenlabs     # with ElevenLabs voice
    python voice_loop.py --face eve --text-only          # type instead of talk (no mic/STT)
    python voice_loop.py --face eve --barge-in           # interrupt it by talking (use headphones)

Timeline of one turn (all stages overlap as much as they can):
    you stop talking ─┐
                      ├─ STT final (Vosk: ~0.2 s after silence)
                      ├─ Claude first token (~0.5-1 s)
                      ├─ first sentence complete -> TTS starts
                      └─ first audio (~0.5 s edge / ~0.2 s ElevenLabs)   face is talking
Every turn prints these numbers so you can see where time goes.

Half-duplex by default: while the face is speaking the mic is ignored, so the
speaker output does not get transcribed as a new question. --barge-in listens
during playback and interrupts when you start talking; that needs headphones
or a mic that does not hear the speakers.
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import re
import sys
import threading
import time
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

from .stt_backends import STTBackend, Transcript


class Speaker:
    """What the loop needs from the face app (TalkerApp satisfies this)."""
    def speak_stream(self, chunks: Iterable[str]) -> None: ...
    def interrupt(self) -> None: ...
    @property
    def is_busy(self) -> bool: ...


class VoiceLoop:
    """
    Feed mic audio to process(); it runs STT, and when an utterance ends it
    streams the LLM reply into the speaker on a worker thread.

    llm_reply(user_text) must return an iterator of text chunks.
    """

    GRACE_AFTER_SPEECH = 0.35      # seconds to keep ignoring the mic after playback ends
    END_MARKER = "[end]"           # the brain appends this when the conversation is over
    DEFAULT_SLEEP_WORDS = ["stop", "wait", "hold on", "hang on", "pause", "quiet", "be quiet", "shush",
                           "go to sleep", "sleep now", "that's enough", "enough", "stop talking",
                           "hold that thought", "one moment", "goodbye", "bye bye", "bye"]

    def __init__(self, stt: STTBackend, llm_reply: Callable[[str], Iterator[str]],
                 speaker: Speaker, barge_in: bool = False,
                 on_event: Optional[Callable[[str, str], None]] = None,
                 wake_words: Optional[List[str]] = None, idle_timeout: float = 60.0,
                 clock=time.monotonic, start_engaged: bool = True,
                 sleep_words: Optional[List[str]] = None,
                 barge_in_ms: int = 400, barge_in_boost: float = 4.0):
        self.stt = stt
        self.llm_reply = llm_reply
        self.speaker = speaker
        self.barge_in = barge_in
        # Barge-in only after `barge_in_ms` of continuous speech, detected with the
        # onset threshold multiplied by `barge_in_boost` while the character talks,
        # so its own voice reaching the mic, or a cough, does not cut it off.
        self.barge_in_ms = barge_in_ms
        self.barge_in_boost = barge_in_boost
        self.barge_in_duty = 0.4       # share of the window the mic must be above the boosted onset level
                                       # (a voice with its syllable gaps: ~0.5; a knock and its ring: ~0.2)
        self._loud: "collections.deque[tuple]" = collections.deque()   # (time, rms) over the barge window
        self.paused = False
        self.waiting = False           # waiting mode (spacebar / control page): ignore the mic and wake words entirely
        self._barge_since: Optional[float] = None
        self._gated = False
        from .audio_engine import EchoGuard
        self._echo = EchoGuard()
        self.echo_threshold = 0.5      # mic/speaker envelope correlation above this = the character's own voice
        self.on_event = on_event or (lambda kind, text: print(f"[{kind}] {text}"))
        self.clock = clock
        # Wake mode: with wake words set, nothing is answered until one is heard;
        # after `idle_timeout` s of silence, or when the brain ends the chat, go dormant again.
        # longest phrases first so "hey eve" wins over "eve"
        self.wake_words = sorted({w.strip().lower() for w in (wake_words or []) if w.strip()},
                                 key=lambda w: (-len(w.split()), -len(w)))
        self.idle_timeout = idle_timeout
        # Start engaged (first visitor need not say the name); go dormant after the
        # idle timeout or a goodbye. start_engaged=False for a kiosk that waits.
        self.engaged = (not self.wake_words) or start_engaged
        # Short utterances that put the character to sleep at once (and cut it off).
        self.sleep_words = [w.strip().lower() for w in (sleep_words if sleep_words is not None
                                                        else self.DEFAULT_SLEEP_WORDS) if w.strip()]
        self._last_activity = self.clock()
        self._lock = threading.Lock()
        self._thinking = False
        self._last_busy = 0.0
        self.on_reply_start: Callable[[], None] = lambda: None   # e.g. hush the ambience
        self._partial = ""
        self.turns = 0
        # Recognition runs on its own thread: the mic callback must return in
        # microseconds or PortAudio drops audio while Whisper is busy.
        self._audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._worker = threading.Thread(target=self._drain, name="stt-worker", daemon=True)
        self._worker.start()
        self._last_audio_in = 0.0
        self._speech_end_at = 0.0     # when the STT said the utterance ended
        # Optional vision.SceneWatcher: a burst is requested the moment the visitor
        # starts talking so the note is fresh when the transcript lands.
        self.vision = None
        self.add_context: Callable[[str], None] = lambda text: None   # wired to the brain
        self._was_speaking = False
        self._vision_ticket: Optional[int] = None
        self._last_heard = 0.0         # last time a partial transcript arrived (someone is talking)
        # Dictation: keep transcribing phrase by phrase but hold the reply until the
        # speaker has been quiet for `dictation_pause_s`; then answer everything at once.
        self.dictating = False
        self.dictation_pause_s = 4.0
        self.dictation_words: List[str] = []       # short utterances that switch it on
        self.dictation_end_words: List[str] = []   # ... and off (the buffer is answered)
        self._dictation: List[str] = []
        self._dictation_last = 0.0
        self.on_dictation: Callable[[bool], None] = lambda on: None    # e.g. say "listening"
        # Session end: the brain writes itself a note (see memory_notes.py).
        self.on_session_end: Callable[[str], None] = lambda reason: None
        self._session_turns = 0

    # ── mic path ────────────────────────────────────────
    def process(self, pcm: bytes) -> None:
        """Called from the audio callback: hand off and return immediately."""
        try:
            self._audio_q.put_nowait(pcm)
        except queue.Full:
            pass                      # recognizer is far behind; drop rather than block the mic

    def _drain(self) -> None:
        while True:
            pcm = self._audio_q.get()
            try:
                self._process(pcm)
            except Exception as e:
                self.on_event("error", f"STT failed: {e}")

    def _process(self, pcm: bytes) -> None:
        self._last_audio_in = time.monotonic()
        if self.paused or self.waiting:   # calibration owns the mic, or waiting mode: hear nothing
            self.stt.reset()
            self._barge_since = None
            return
        self.tick()
        busy = self.speaker.is_busy or self._thinking
        now = self.clock()
        if busy:
            self._last_busy = now
            if not self.barge_in or self._thinking:
                self.stt.reset()          # drop echo; nothing to transcribe
                self._barge_since = None
                return
            if not self._gated and hasattr(self.stt, "set_playback_gate"):
                self.stt.set_playback_gate(self.barge_in_boost)
                self._gated = True
            t = self.stt.feed(pcm)
            import numpy as _np
            frame = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32)
            self._echo.add_mic(now, float(_np.sqrt(_np.mean(frame * frame))) if frame.size else 0.0)
            rms = float(_np.sqrt(_np.mean(frame * frame))) if frame.size else 0.0
            self._loud.append((now, rms))
            while self._loud and now - self._loud[0][0] > self.barge_in_ms / 1000.0:
                self._loud.popleft()
            talking = bool(getattr(self.stt, "speech_active", False)) or bool(t and t.text)
            if talking:
                if self._barge_since is None:
                    self._barge_since = now
                elif (now - self._barge_since) * 1000 >= self.barge_in_ms and self._sustained():
                    env = getattr(self.speaker, "output_envelope", None)
                    corr = self._echo.correlation(env(), now) if env else 0.0
                    if corr >= self.echo_threshold:
                        self.on_event("echo", f"mic follows the speaker (corr {corr:.2f}); not a barge-in")
                        self._barge_since = None          # start over; a person will break the pattern
                        return
                    self.on_event("barge-in", t.text if t and t.text else f"speech for {self.barge_in_ms} ms (corr {corr:.2f})")
                    self.speaker.interrupt()
                    self._last_busy = 0.0
                    self._barge_since = None
            else:
                self._barge_since = None
            return
        if self._gated and hasattr(self.stt, "set_playback_gate"):
            self.stt.set_playback_gate(1.0)
            self._gated = False
        self._barge_since = None
        if now - self._last_busy < self.GRACE_AFTER_SPEECH:
            self.stt.reset()
            return

        t = self.stt.feed(pcm)
        speaking = bool(getattr(self.stt, "speech_active", False))
        if speaking and not self._was_speaking and self.vision is not None:
            # look now, but still only pay for a description if the picture changed
            self._vision_ticket = self.vision.request(force=False)
        self._was_speaking = speaking
        if t is None:
            return
        if speaking and self.engaged:
            self._last_activity = self.clock()
        if not t.final:
            if t.text != self._partial:
                self._partial = t.text
                self._last_heard = time.monotonic()
                self.on_event("hearing", t.text)
            return
        self._partial = ""
        if t.text.strip():
            self._last_heard = time.monotonic()
            self._speech_end_at = (time.monotonic() - getattr(self.stt, "endpoint_delay_s", 0.0)
                                   - getattr(self.stt, "last_transcribe_s", 0.0))
            self.on_user_text(t.text.strip())

    # ── wake mode ───────────────────────────────────────
    def _sustained(self) -> bool:
        """True when the mic stayed loud for most of the barge window. The endpointer's
        'active' flag lingers through its silence gate, so a single knock would otherwise
        count as continuous speech."""
        thr = self.stt.onset_threshold() if hasattr(self.stt, "onset_threshold") else None
        if not thr or len(self._loud) < 3:
            return True
        loud = sum(1 for _, r in self._loud if r >= thr)
        return loud / len(self._loud) >= self.barge_in_duty

    @staticmethod
    def _norm(text: str) -> List[str]:
        return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()

    def find_wake_word(self, text: str) -> Optional[Tuple[int, int]]:
        """(start, end) token span of a wake word in text, tolerant of small mis-hearings."""
        import difflib
        words = self._norm(text)
        for wake in self.wake_words:
            wtoks = wake.split()
            n = len(wtoks)
            for i in range(len(words) - n + 1):
                window = words[i:i + n]
                if window == wtoks:
                    return (i, i + n)
                # Names get mangled by STT; allow near matches for longer names only
                # (short ones like "goat"/"coat" collide too easily: list variants instead).
                if n == 1 and len(wtoks[0]) >= 5 and difflib.SequenceMatcher(None, window[0], wtoks[0]).ratio() >= 0.8:
                    return (i, i + 1)
                if n > 1 and difflib.SequenceMatcher(None, " ".join(window), wake).ratio() >= 0.85:
                    return (i, i + n)
        return None

    def is_sleep_command(self, text: str) -> bool:
        """A short utterance that is (mostly) a sleep word: 'stop', 'hold on', 'wait a sec'."""
        if not self.wake_words or not self.sleep_words:
            return False
        words = self._norm(text)
        if not words or len(words) > 4:
            return False
        joined = " ".join(words)
        for phrase in self.sleep_words:
            if joined == phrase or joined.startswith(phrase + " ") or joined.endswith(" " + phrase):
                return True
        return False

    def engage(self, reason: str = "") -> None:
        if not self.engaged:
            self.engaged = True
            self.on_event("mode", f"engaged{(' (' + reason + ')') if reason else ''}")
        self._last_activity = self.clock()

    def disengage(self, reason: str = "") -> None:
        if self.engaged and self.wake_words:
            self.engaged = False
            self.on_event("mode", f"dormant, listening for {', '.join(repr(w) for w in self.wake_words)}"
                                  f"{(' (' + reason + ')') if reason else ''}")
            self.end_session(reason)

    def end_session(self, reason: str = "") -> None:
        """The conversation is over (dormant, goodbye, window closed): give the memory a
        chance to write its summary. Fires once per stretch of conversation."""
        turns, self._session_turns = self._session_turns, 0
        if turns < 2:
            return
        try:
            self.on_session_end(reason)
        except Exception as e:
            print(f"[loop] session end: {e}")

    def tick(self) -> None:
        """Idle timeout check; called per audio chunk. Silence is counted from the end of the
        character's own speech, so a long reply never eats into the timeout."""
        if (self.dictating and self._dictation and not self._partial
                and self.clock() - self._dictation_last >= self.dictation_pause_s):
            self._flush_dictation("pause")
        if not (self.engaged and self.wake_words):
            return
        if self.speaker.is_busy or self._thinking:
            self._last_activity = self.clock()
            return
        if self.clock() - self._last_activity > self.idle_timeout:
            self.disengage(f"quiet for {self.idle_timeout:.0f}s")

    # ── one turn ────────────────────────────────────────
    def set_waiting(self, on: bool, reason: str = "") -> None:
        """Waiting mode: stop talking, ignore the mic, wake words and typed text until
        switched back. Nothing automatic leaves it. The engaged state is kept for later."""
        on = bool(on)
        if on == self.waiting:
            return
        self.waiting = on
        if on:
            self.speaker.interrupt()
            self._thinking = False
        self.on_event("mode", ("waiting" if on else "listening again") + (f" ({reason})" if reason else ""))

    def on_user_text(self, text: str) -> None:
        """Handle a finished user utterance (also used by --text-only)."""
        if self.waiting:
            self.on_event("ignored", f"waiting mode: {text}")
            return
        if self.wake_words:
            if self.engaged and self.is_sleep_command(text):
                self.on_event("you", text)
                self.speaker.interrupt()
                self.disengage("sleep word")
                return
            span = self.find_wake_word(text)
            if not self.engaged:
                if span is None:
                    self.on_event("ignored", text)          # dormant: not for us
                    return
                words = self._norm(text)
                rest = " ".join(words[:span[0]] + words[span[1]:]).strip()
                self.engage("heard the wake word")
                text = rest if rest else f"{self.wake_words[-1].title()}?"     # shortest = the name
            else:
                self._last_activity = self.clock()
        if self._is_phrase(text, self.dictation_end_words) and self.dictating:
            self.set_dictation(False, "end word")
            return
        if self._is_phrase(text, self.dictation_words) and not self.dictating:
            self.set_dictation(True, "spoken")
            return
        if self.dictating:
            self._dictation.append(text)
            self._dictation_last = self.clock()
            self.on_event("dictation", " ".join(self._dictation))
            return
        self._start_turn(text)

    def _start_turn(self, text: str) -> None:
        with self._lock:
            if self._thinking:
                return
            self._thinking = True
        self.turns += 1
        self._session_turns += 1
        self.on_event("you", text)
        threading.Thread(target=self._answer, args=(text,), daemon=True, name="llm-turn").start()

    # ── dictation ───────────────────────────────────────
    def _is_phrase(self, text: str, phrases: List[str]) -> bool:
        """A short utterance that is one of the phrases (the sleep-word rule)."""
        if not phrases:
            return False
        words = self._norm(text)
        if not words or len(words) > 5:
            return False
        joined = " ".join(words)
        return any(joined == p or joined.startswith(p + " ") or joined.endswith(" " + p)
                   for p in phrases)

    def set_dictation(self, on: bool, reason: str = "") -> None:
        """Dictation on: phrases are collected and answered together after a long pause.
        Off: whatever was collected is answered now."""
        on = bool(on)
        if on == self.dictating:
            return
        self.dictating = on
        self.on_event("mode", ("dictation: answering after each "
                               f"{self.dictation_pause_s:.0f} s pause" if on else "dictation off")
                      + (f" ({reason})" if reason else ""))
        try:
            self.on_dictation(on)
        except Exception as e:
            print(f"[loop] on_dictation: {e}")
        if not on and self._dictation:
            self._flush_dictation(reason or "off")

    def _flush_dictation(self, why: str) -> None:
        text = " ".join(self._dictation).strip()
        self._dictation = []
        if text:
            self._start_turn(text)

    # ── a turn nobody asked for ─────────────────────────
    def announce(self, event_text: str) -> bool:
        """Tell the brain something happened and let it speak up, but only when the room is
        quiet: not while it talks or thinks, not while someone is mid-sentence, not within
        a couple of seconds of either. Returns False to say: try again later."""
        if self.waiting or self.paused or not self.engaged or self.dictating:
            return False
        if self.speaker.is_busy or self._thinking or self._partial:
            return False
        now = time.monotonic()
        if now - self._last_heard < 2.0 or self.clock() - self._last_busy < 2.0:
            return False
        with self._lock:
            if self._thinking:
                return False
            self._thinking = True
        self.turns += 1
        self._session_turns += 1
        self._speech_end_at = 0.0                    # no utterance to time this against
        self.on_event("event", event_text)
        threading.Thread(target=self._answer, args=(f"(Event: {event_text})",),
                         daemon=True, name="llm-turn").start()
        return True

    def _strip_end_marker(self, chunks: Iterator[str]) -> Iterator[str]:
        """Remove [end] from the stream (it may straddle chunks) and disengage if seen."""
        buf = ""
        keep = len(self.END_MARKER) - 1
        for chunk in chunks:
            buf += chunk
            if self.END_MARKER in buf.lower():
                idx = buf.lower().index(self.END_MARKER)
                out, buf = buf[:idx], buf[idx + len(self.END_MARKER):]
                self._ended = True
                if out:
                    yield out
                continue
            if len(buf) > keep:
                yield buf[:-keep]
                buf = buf[-keep:]
        if buf:
            yield buf

    VISUAL_RE = re.compile(r"\b(see|look|watch|holding|hold|wearing|wear|this|that|these|expression|"
                           r"face|colou?r|what am i|who am i|how many|show|showing|picture|drawing)\b", re.I)

    def _answer(self, text: str) -> None:
        self._ended = False
        try:
            self.on_reply_start()
        except Exception as e:
            print(f"[loop] on_reply_start: {e}")
        # A visual question: give the in-flight burst time to land first (capture ~0.5 s
        # + vision model ~2.3 s, minus what already elapsed while the visitor spoke).
        if self.vision is not None and self.VISUAL_RE.search(text):
            seen = self.vision.look_now(timeout=2.5)
            if seen:
                self.add_context(f"right now you can see: {seen}")
        self._vision_ticket = None
        t_end = time.monotonic()
        t_stop = self._speech_end_at or t_end       # typed text: no STT stage
        stats = {"first_token_ms": None}
        first_audio_before = getattr(self.speaker, "first_audio_at", 0.0)

        def report_when_audio_starts():
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                fa = getattr(self.speaker, "first_audio_at", 0.0)
                if fa and fa != first_audio_before and fa >= t_end:
                    tok = stats["first_token_ms"]
                    fs = getattr(self.speaker, "first_sentence_at", 0.0)
                    sent = f" -> first sentence to TTS {round((fs - t_stop) * 1000)} ms" if fs and fs >= t_end else ""
                    self.on_event("turn", f"you stopped -> transcript {round((t_end - t_stop) * 1000)} ms"
                                  f" -> first token {tok if tok is not None else '?'} ms{sent}"
                                  f" -> first audio {round((fa - t_stop) * 1000)} ms")
                    return
                time.sleep(0.02)
        threading.Thread(target=report_when_audio_starts, daemon=True).start()

        def timed_chunks():
            reply = []
            try:
                for chunk in self._strip_end_marker(self.llm_reply(text)):
                    if stats["first_token_ms"] is None:
                        stats["first_token_ms"] = round((time.monotonic() - t_stop) * 1000)
                        self._thinking = False        # speaker is now busy; mic stays gated
                    reply.append(chunk)
                    yield chunk
            except Exception as e:
                self.on_event("error", f"LLM failed: {e}")
                yield "[sad]Sorry, I could not think of an answer just now."
            finally:
                self._thinking = False
                self._last_activity = self.clock()
                self.on_event("bot", "".join(reply).strip())
                if self._ended:
                    self.disengage("the conversation ended")


        self.speaker.speak_stream(timed_chunks())


# ═══════════════════════════════════════════════════════
# PROFILES
# ═══════════════════════════════════════════════════════
def apply_pi_profile(args) -> None:
    """
    Raspberry Pi: keep every heavy stage in the cloud. Local Whisper takes
    seconds per turn there; ElevenLabs Scribe realtime commits ~0.5 s after
    you stop with no local CPU. Only fills in what the user did not set
    explicitly.
    """
    if args.stt == "whisper":          # the parser default, i.e. not chosen by the user
        args.stt = "elevenlabs"
    if args.silence_ms is None:
        args.silence_ms = 500
    args.fullscreen = True
    args.thinking = False
    args.wake = True                       # a kiosk waits to be called by name
    args.start_dormant = True
    os.environ.setdefault("TALKER_FPS", "30")
    print(f"[profile] pi: stt={args.stt} llm={args.llm} fullscreen, 30 fps")


# ═══════════════════════════════════════════════════════
# MIC TOOLS  (--list-devices / --mic-test)
# ═══════════════════════════════════════════════════════
def stt_kwargs(args) -> dict:
    if args.stt == "vosk":
        return {"model_path": args.vosk_model, "silence_ms": args.silence_ms}
    if args.stt == "whisper":
        return {"model_size": args.whisper_model, "silence_ms": args.silence_ms}
    if args.stt in ("elevenlabs", "openai"):
        return {"silence_ms": args.silence_ms}
    return {}


def mic_tools(args) -> int:
    from .audio_engine import AudioEngine
    audio = AudioEngine(output_device=args.output_device)
    print("Input devices:")
    for idx, name, rate, is_default in audio.list_input_devices():
        print(f"  [{idx}] {name}  ({rate} Hz){'  <- default' if is_default else ''}")
    if args.list_devices:
        print("Output devices:")
        for idx, name, rate, is_default in audio.list_output_devices():
            print(f"  [{idx}] {name}  ({rate} Hz){'  <- default' if is_default else ''}")
        audio.close()
        return 0

    from .stt_backends import make_stt, EnergyEndpointer
    stt = make_stt(args.stt, **stt_kwargs(args))
    state = {"peak": 0.0, "last": "", "n": 0}
    recorder = None
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        recorder = EnergyEndpointer(stt.sample_rate, silence_ms=args.silence_ms or 600)
        print(f"Recording utterances to {args.record}/ (WAV + transcripts.txt draft references)")

    def save_clip(audio: bytes, text: str):
        import wave
        state["n"] += 1
        name = f"utt_{state['n']:03d}.wav"
        with wave.open(os.path.join(args.record, name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(stt.sample_rate); w.writeframes(audio)
        with open(os.path.join(args.record, "transcripts.txt"), "a", encoding="utf-8") as f:
            f.write(f"{name}\t{text}\n")
        print(f"[saved]   {name} ({len(audio)/2/stt.sample_rate:.1f}s)")

    pending_clip = {"audio": None}

    def on_frames(pcm):
        import numpy as np
        s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
        state["peak"] = max(state["peak"], rms)
        if args.calibrate:
            return                                  # levels only; no transcripts during calibration
        if recorder is not None:
            clip = recorder.feed(pcm)
            if clip is not None:
                pending_clip["audio"] = clip
        t = stt.feed(pcm)
        if t and t.final:
            print(f"\n[final]   {t.text}")
            if recorder is not None and pending_clip["audio"] is not None:
                save_clip(pending_clip["audio"], t.text)
                pending_clip["audio"] = None
        elif t and t.text != state["last"]:
            state["last"] = t.text
            print(f"\r[hearing] {t.text[-70:]:<70}", end="", flush=True)

    audio.start_mic(on_frames=on_frames, rate=stt.sample_rate, device=args.mic_device, open_rate=args.mic_rate)
    pipeline = None
    if args.calibrate:
        from .calibrate import run_calibration
        from .phoneme_scheduler import ScheduleReader
        from .speech_pipeline import SpeechPipeline
        from .tts_backends import make_backend
        tts = (args.tts or ("elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "edge")).lower()
        pipeline = SpeechPipeline(audio, ScheduleReader(), make_backend(tts, voice=args.voice, model=args.tts_model))
        pipeline.start()
        try:
            run_calibration(audio, pipeline.speak, lambda: pipeline.is_busy, args.mic_device, args.output_device)
        finally:
            pipeline.stop()
            audio.close()
        return 0
    if args.play:
        from .phoneme_scheduler import ScheduleReader
        from .speech_pipeline import SpeechPipeline
        from .tts_backends import make_backend
        tts = (args.tts or ("elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "edge")).lower()
        pipeline = SpeechPipeline(audio, ScheduleReader(), make_backend(tts, voice=args.voice, model=args.tts_model))
        pipeline.start()
        pipeline.speak(" ".join([args.play] * 3))
        print("Playing the character's voice through the speaker: the level you see now is what the mic "
              "hears FROM THE SPEAKER. Then talk from the visitor spot and compare.")
    print("Speak. Level is printed every 2 s (aim for 500-5000; below ~200 is too quiet). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(2)
            rms, _ = audio.get_state()
            tag = "speaker" if pipeline is not None and pipeline.is_busy else "mic    "
            print(f"\r[level {tag}] now {rms:5.0f}  peak {state['peak']:5.0f}{' ':46}")
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline is not None:
            pipeline.stop()
        audio.close()
    return 0


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Talk to an animated face: mic -> STT -> Claude -> voice")
    p.add_argument("--profile", choices=["desktop", "pi"], default=None,
                   help="pi: cloud speech-to-text (elevenlabs), fullscreen, 30 fps — nothing heavy "
                        "runs locally. Explicit flags still win.")
    p.add_argument("--face", default="eve")
    p.add_argument("--face-dir", default=None)
    p.add_argument("--stt", default="whisper",
                   help="whisper (default, local, accurate) | vosk (local, light) | "
                        "elevenlabs (cloud Scribe realtime, best for a Pi) | openai (cloud batch)")
    p.add_argument("--vosk-model", default=None,
                   help="small (default) | lgraph | large | path. Larger = more accurate")
    p.add_argument("--whisper-model", default=None, help="faster-whisper size, e.g. base.en, small.en")
    p.add_argument("--silence-ms", type=int, default=None,
                   help="Silence that ends an utterance (default 600). Lower = snappier, more mid-sentence cuts")
    p.add_argument("--mic-device", default=None, help="Input device: name fragment (\"samson\") or index")
    p.add_argument("--mic-rate", type=int, default=None, help="Force the device to open at this rate")
    p.add_argument("--output-device", default=None, help="Output device: name fragment (\"jabra\") or index")
    p.add_argument("--list-devices", action="store_true", help="List input and output devices and exit")
    p.add_argument("--calibrate", action="store_true",
                   help="Guided check of a mic/speaker setup: room, speaker bleed, a person; writes calibration.json "
                        "whose values become the defaults for --mic-device/--output-device/--barge-in-boost")
    p.add_argument("--mic-test", action="store_true",
                   help="Only print what the mic hears (levels + transcripts); no Claude, no voice")
    p.add_argument("--play", default=None, metavar="TEXT",
                   help="With --mic-test: also speak TEXT through the output device (repeats 3 times) so you "
                        "can read the mic level from the speaker versus from a person; set --tts/--voice as usual")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="With --mic-test: save each utterance as WAV in DIR plus transcripts.txt "
                        "(draft references to correct, then run tools/bench_stt.py DIR)")
    p.add_argument("--tts", default=None,
                   help="elevenlabs (default when ELEVENLABS_API_KEY is set), piper (local, offline), fish (Fish Audio) or edge (free)")
    p.add_argument("--voice", default=None, help="TTS voice name/id")
    p.add_argument("--voice-speed", type=float, default=None,
                   help="Speaking rate multiplier, e.g. 1.15 (ElevenLabs Flash and edge honour it; v3 ignores it)")
    p.add_argument("--tts-model", default=None,
                   help="ElevenLabs model: v3 (default; performs [sigh]/[excited]-style tags, ~1 s to first audio) "
                        "or flash (~0.25 s, tags stripped). Full model ids also accepted.")
    p.add_argument("--llm", default="claude", choices=["claude", "openai", "ollama"],
                   help="Which brain answers: claude (default), openai (for comparison), or ollama (local, no key)")
    p.add_argument("--ollama-host", default=None,
                   help="Ollama server URL for --llm ollama (default OLLAMA_HOST or http://localhost:11434)")
    p.add_argument("--list-models", action="store_true", help="List the models on the Ollama server and exit")
    p.add_argument("--model", default=None,
                   help="Model id for the chosen --llm (defaults: claude-haiku-4-5 for speed; "
                        "--model claude-opus-5 for the best writing at ~2 s more per reply; openai: gpt-4o-mini)")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--thinking", action="store_true",
                   help="Enable Claude's reasoning pass before answering (about +1 s to first token; off by default)")
    p.add_argument("--no-thinking", action="store_true", help=argparse.SUPPRESS)   # kept for old scripts
    p.add_argument("--character", default=None, help='Persona, e.g. "EVE from WALL-E, terse and curious"')
    p.add_argument("--barge-in", action="store_true", help="Interrupt playback when you start talking")
    p.add_argument("--barge-in-ms", type=int, default=400,
                   help="Continuous speech needed before a barge-in interrupts (default 400 ms, which is about two syllables; a cough or a clatter is shorter than that, and the loudness check catches the rest)")
    p.add_argument("--barge-in-boost", type=float, default=None,
                   help="How much louder than usual speech must be, while the character talks, to count "
                        "(multiplier on the onset threshold; default 4; raise if it still cuts itself off, lower to 2.5 for a quiet room)")
    p.add_argument("--text-only", action="store_true", help="Type in the window instead of using the mic")
    p.add_argument("--debug", "-d", action="store_true")
    p.add_argument("--no-hud", action="store_true", help="Hide key hints and text box")
    p.add_argument("--fullscreen", action="store_true",
                   help="Projection mode: fullscreen, face scaled to the display, no overlay or cursor (F toggles)")
    p.add_argument("--borderless", action="store_true",
                   help="Projection as a frameless desktop-sized window instead of exclusive fullscreen: no display "
                        "mode switch, no compositor artifacts; the face is scaled in software (about 1 ms a frame)")
    p.add_argument("--sync-offset", type=float, default=0.0)
    p.add_argument("--no-audio", action="store_true", help="No sound device (implies --text-only)")
    p.add_argument("--wake", action="store_true",
                   help="Wake mode: stay dormant until the character's name (or --wake-word) is heard; "
                        "go dormant again after --idle-timeout seconds of silence or when the chat ends")
    p.add_argument("--wake-word", default=None,
                   help='Comma-separated wake words (implies --wake), e.g. "eve, hey eve". '
                        "Default: the face's wake_words, else its name")
    p.add_argument("--idle-timeout", type=float, default=60.0,
                   help="Seconds of silence after the character last spoke before it goes dormant (default 60)")
    p.add_argument("--start-dormant", action="store_true",
                   help="In wake mode, start dormant instead of engaged (kiosk: wait to be called by name)")
    p.add_argument("--sleep-word", default=None,
                   help='Comma-separated phrases that put the character to sleep at once (default: "stop, wait, '
                        'hold on, hang on, pause, quiet, go to sleep, enough, goodbye, bye" and a few more)')
    p.add_argument("--camera", default=None,
                   help='Turn on vision: camera name fragment ("c920") or index. Off unless given.')
    p.add_argument("--no-vision", action="store_true", help="Force vision off even if a profile or face enables it")
    p.add_argument("--list-cameras", action="store_true", help="List cameras and exit")
    p.add_argument("--vision-interval", type=float, default=9.0, help="Seconds between camera bursts (default 9)")
    p.add_argument("--vision-frames", type=int, default=3, help="Frames per burst (default 3)")
    p.add_argument("--vision-model", default="claude-haiku-4-5", help="Vision model for scene notes")
    p.add_argument("--vision-change", type=float, default=0.035,
                   help="Change filter: a burst is sent to the vision model only if the picture differs from "
                        "the last described one by more than this fraction (default 0.035 = 3.5%% mean pixel "
                        "change; a still room is ~1%%, a person entering 10%%+). A refresh is forced every 90 s.")
    p.add_argument("--fixed-fps", action="store_true",
                   help="Disable the adaptive frame rate (default: step down to 45/30/20/15 fps under load, recover later)")
    p.add_argument("--web-port", type=int, default=8020,
                   help="Control page on this port (status, live tuning, setup tests, flag reference, Wi-Fi); default 8020, the next free port if taken")
    p.add_argument("--setup", action="store_true",
                   help="Guided first-run setup: checks packages and ffmpeg, validates your API keys against the services, picks and tests the speaker, microphone and camera, measures the room, and prints the command to run")
    p.add_argument("--mic-highpass", type=float, default=90.0,
                   help="High-pass the microphone at this many Hz before anything measures the "
                        "level: cuts hum, air conditioning, traffic and desk thumps, which carry no "
                        "words but do move the speech gate. 0 disables it (default 90)")
    p.add_argument("--mic-lowpass", type=float, default=7500.0,
                   help="Low-pass the microphone at this many Hz before it is resampled down for "
                        "the recognizer. This is anti-aliasing: without it, noise above half the "
                        "recognizer's rate folds back into the speech band. 0 disables it (default 7500)")
    p.add_argument("--web-host", default="127.0.0.1",
                   help="Interface for the control page; default 127.0.0.1, this machine only. "
                        "0.0.0.0 opens it to the network, which a kiosk wants, but the page has no "
                        "password and serves a camera snapshot and a Wi-Fi join endpoint")
    p.add_argument("--no-web", action="store_true", help="Do not start the control page")
    p.add_argument("--agent-url", default=None,
                   help="Backend agent that {{task ...}} errands go to (tools/agent_relay.py on the machine "
                        "with your agent harness), e.g. http://agentbox:8030; default AGENT_RELAY_URL from "
                        ".env. Only a face with \"errands\": true uses it")
    p.add_argument("--errand-poll", type=float, default=5.0,
                   help="Seconds between polls of the backend agent for finished tasks (default 5)")
    p.add_argument("--dictation-pause", type=float, default=4.0,
                   help="Dictation mode: seconds of quiet before everything you dictated is answered as one "
                        "turn (default 4). Tab in the window or a face's dictation_words switch it on")
    return p


def main(argv=None) -> int:
    from .env_config import load_dotenv
    load_dotenv()
    from .session_log import start_session_log
    log_path = start_session_log("voice")
    p = build_parser()
    args = p.parse_args(argv)
    if args.profile == "pi":
        apply_pi_profile(args)
    # calibration.json (from --calibrate) supplies defaults for what was not given explicitly
    from .calibrate import load_calibration
    cal = load_calibration()
    if cal:
        used = []
        # A saved device may simply be unplugged today. That is a reason to fall back to
        # the system default with a warning, not a reason to refuse to start: the file is
        # a convenience, and nothing in it was typed on this run.
        def still_there(name: str, kind: str) -> bool:
            try:
                from .audio_engine import AudioEngine
                probe = AudioEngine()
                try:
                    probe.resolve_devices(name, kind)
                    return True
                finally:
                    probe.close()
            except Exception:
                print(f"[calibration] {kind} device {name!r} from calibration.json is not here "
                      f"any more; using the system default instead")
                return False

        if args.mic_device is None and cal.get("mic_device") and still_there(cal["mic_device"], "input"):
            args.mic_device = cal["mic_device"]; used.append(f"mic {cal['mic_device']}")
        if args.output_device is None and cal.get("output_device") and still_there(cal["output_device"], "output"):
            args.output_device = cal["output_device"]; used.append(f"output {cal['output_device']}")
        if args.barge_in_boost is None and cal.get("barge_in_boost"):
            args.barge_in_boost = float(cal["barge_in_boost"]); used.append(f"barge-in boost {cal['barge_in_boost']}")
        if used:
            print(f"[calibration] using {', '.join(used)} from calibration.json ({cal.get('time', '')})")
        # The file records more than the three values above. Say the awkward parts out
        # loud rather than leaving them to be read out of a JSON file nobody opens.
        if args.barge_in and cal.get("barge_in_ok") is False:
            print(f"[calibration] warning: --barge-in was measured as unworkable here. "
                  f"{cal.get('verdict', '')}")
        if cal.get("mic_gain") == "raise":
            print("[calibration] warning: the microphone measured too quiet. Raise its gain in the "
                  "system sound settings, or speech will be missed.")
        elif cal.get("mic_gain") == "lower":
            print("[calibration] warning: the microphone measured hot enough to clip. Lower its gain.")
    if args.barge_in_boost is None:
        args.barge_in_boost = 4.0

    import pygame
    from .face_asset_loader import FaceAssetLoader, default_manifest
    from .app import TalkerApp, build_audio, resolve_face_dir
    from .tts_backends import make_backend
    from .brains.claude_chat import ClaudeChat

    if args.list_models:
        from .brains.ollama_chat import list_models, DEFAULT_HOST
        host = args.ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
        try:
            names = list_models(host)
        except Exception as e:
            print(f"Ollama at {host} not reachable: {e}")
            return 1
        print(f"Models on {host}:")
        for n in names:
            print(f"  {n}")
        print("Use one with: --llm ollama --model <name>")
        return 0
    if args.list_cameras:
        from .vision import list_cameras
        cams = list_cameras()
        print("Cameras:" if cams else "No cameras found")
        for c in cams:
            print(f"  [{c['index']}] {c['name']}  ({c['path']})")
        return 0
    if args.setup:
        from .setup_wizard import run_setup
        return run_setup(args)
    if args.list_devices or args.mic_test or args.calibrate:
        return mic_tools(args)

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)
    face_dir = resolve_face_dir(args.face, args.face_dir)
    # An assistant's memory (notes.md, tasks.md next to face.json) reaches character.md
    # through {notes} and {tasks}, filled while the face loads: register them first.
    notebook = ledger = None
    if face_dir:
        from .memory_notes import Notebook, TaskLedger
        from .launch_facts import register
        notebook, ledger = Notebook(face_dir), TaskLedger(face_dir)
        register("notes", notebook.recent)
        register("tasks", ledger.render)
    assets = FaceAssetLoader().load(face_dir) if face_dir else FaceAssetLoader().build(default_manifest(args.face))

    # Face-level defaults for voice, model and persona (flags win)
    m = assets.manifest
    args.tts = (args.tts or m.tts or            # face.json "tts" picks this face's own backend
                ("elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "edge")).lower()
    voice = args.voice or m.voices.get(args.tts)
    tts_model = args.tts_model or m.tts_model or ("eleven_v3" if args.tts == "elevenlabs" else None)
    character = args.character or m.character or None
    speed = args.voice_speed or m.voice_speed

    # Actions the brain may write next to its speech ({{move nod}}, {{sfx creak}},
    # {{tool ...}}); it is only told about the ones this face has.
    from .actions import ActionDispatcher, SoundBank, ToolBox, action_rules
    from .body import NullBody
    sounds = SoundBank(os.path.join(face_dir, m.sounds) if face_dir else None)
    body = NullBody((m.body or {}).get("moves", []))
    tools = ToolBox()
    # An assistant: notes she writes herself, and errands for a backend agent (errands.py).
    from .brains.claude_chat import assistant_rules
    memory_on = bool(m.memory) and notebook is not None
    agent_url = (args.agent_url or os.environ.get("AGENT_RELAY_URL") or "").strip()
    errands_on = bool(m.errands)
    if errands_on and not agent_url:
        print("[errands] face.json asks for a backend agent but AGENT_RELAY_URL is not set in .env "
              "and --agent-url was not given: errands off")
        errands_on = False
    if errands_on:
        tools.add("tasks", "the task ledger with each task's state", lambda _a: ledger.render())
    extra_rules = action_rules(body.moves, sounds.names, tools.describe()) + assistant_rules(memory_on, errands_on)
    if extra_rules:
        print(f"[voice] actions: moves={body.moves or '-'} sounds={sounds.names or '-'} tools={tools.names or '-'}"
              + (" note" if memory_on else "") + (" task" if errands_on else ""))
    try:
        backend = make_backend(args.tts, voice=voice, model=tts_model, speed=speed)
    except Exception as e:
        print(f"[error] TTS backend '{args.tts}' unavailable: {e}")
        return 1
    audio = build_audio(args.no_audio, args.sync_offset, args.output_device)
    audio.mic_high_pass = max(0.0, args.mic_highpass)
    audio.mic_low_pass = max(0.0, args.mic_lowpass)
    text_only = args.text_only or args.no_audio

    can_see = args.camera is not None and not args.no_vision
    wake_words = None
    if args.wake or args.wake_word:
        wake_words = ([w for w in args.wake_word.split(",")] if args.wake_word
                      else (m.wake_words or [m.name.replace("_", " ")]))
    try:
        if args.llm == "claude":
            chat = ClaudeChat(model=args.model or "claude-haiku-4-5", effort=args.effort, character=character,
                              thinking=bool(args.thinking), can_see=can_see, wake_mode=bool(wake_words),
                              extra_rules=extra_rules)
        elif args.llm == "ollama":
            from .brains.ollama_chat import OllamaChat
            chat = OllamaChat(model=args.model, character=character, host=args.ollama_host,
                              can_see=can_see, wake_mode=bool(wake_words), extra_rules=extra_rules)
            print(f"[voice] loading {chat.model} on {chat.host} ...")
            chat.warm_up()                   # load the weights now, not on the first question
        else:
            from .brains.openai_compat_chat import OpenAICompatChat
            chat = OpenAICompatChat.openai(model=args.model, character=character, can_see=can_see,
                                           wake_mode=bool(wake_words), extra_rules=extra_rules)
    except Exception as e:                   # a wrong model name or a missing key, said plainly
        print(f"[error] brain '{args.llm}' unavailable: {e}")
        print("        python voice_loop.py --setup checks your keys and the models you have")
        return 1
    print(f"[voice] brain: {args.llm} {chat.model}{' (told it can see)' if can_see else ''}")
    app = TalkerApp(assets, audio, backend, debug=args.debug, show_hud=not args.no_hud,
                    fullscreen=args.fullscreen, adaptive_fps=not args.fixed_fps, borderless=args.borderless)
    actions = ActionDispatcher(on_result=lambda a, r: chat.add_context(f"the {a.name} tool answered: {r}"))
    actions.register("move", body.handler())
    _pipe = getattr(app, "pipeline", None)
    if m.sfx_over_speech or _pipe is None:
        actions.register("sfx", sounds.handler())
    else:                       # default: hold the sound until she has finished the sentence
        actions.register("sfx", sounds.handler(
            hold_while=lambda: _pipe.is_busy,
            ends_at=lambda: _pipe.speech_end_time - audio.timeline_time()))
    actions.register("tool", tools.handler())
    if memory_on:
        def note_handler(a):                       # a file append: microseconds, safe on the speech thread
            line = notebook.note(f"{a.name} {a.args}".strip())
            if line:
                print(f"[note] {line}")
        actions.register("note", note_handler)
    runner = None
    last_you = {"text": ""}
    if errands_on:
        from .errands import ErrandRunner

        def on_started(e):
            ledger.set_state(e.id, "in progress")

        def on_done(e):
            ledger.set_state(e.id, "done", summary=e.summary or e.result[:300])
            runner.say_later(f'Task {e.id} "{e.task}" finished. Result: {e.summary or e.result[:600]}')

        def on_fail(e):
            ledger.set_state(e.id, "failed", summary=e.summary or "failed")
            runner.say_later(f'Task {e.id} "{e.task}" failed: {e.summary or "no reason given"}')

        runner = ErrandRunner(agent_url, poll_s=args.errand_poll, on_started=on_started,
                              on_done=on_done, on_fail=on_fail, sender=m.name)

        def task_handler(a):                       # enqueue only; the poller thread does the HTTP
            task = f"{a.name} {a.args}".strip()
            if not task:
                return None
            e = runner.submit(task, context=f"the person had just said: {last_you['text']}" if last_you["text"] else "")
            ledger.add(e.id, e.task)
            print(f"[task] {e.id} queued: {e.task}")
            return None
        actions.register("task", task_handler)
    if getattr(app, "pipeline", None) is not None:
        app.pipeline.on_action = actions.dispatch

    stt = None
    if not text_only:
        try:
            from .stt_backends import make_stt
            stt = make_stt(args.stt, **stt_kwargs(args))
        except Exception as e:
            print(f"[error] STT backend '{args.stt}' unavailable: {e}")
            return 1

    watcher = None
    if args.camera is not None and not args.no_vision:
        try:
            from .vision import CameraSource, SceneWatcher, describe_with_claude, resolve_camera
            cam_index = resolve_camera(args.camera)
            source = CameraSource(cam_index)
            def on_note(n):
                if args.debug:
                    print(f"[scene] {'EMERGENCY ' if n.emergency else ''}people={n.people}: "
                          f"{n.changes or n.notes}")
                ctx = watcher.context()
                if ctx:
                    chat.add_context(ctx)                    # only fires on a real change

            watcher = SceneWatcher(source, lambda frames, prev: describe_with_claude(frames, prev, model=args.vision_model),
                                   interval=args.vision_interval, burst=args.vision_frames,
                                   change_threshold=args.vision_change, on_note=on_note)
            watcher.start()
            print(f"[vision] on: camera {cam_index}, {args.vision_frames} frames every {args.vision_interval:.0f}s, "
                  f"{args.vision_model}, described only when the scene changes; frames are not stored "
                  f"(emergencies go to emergencies/)")
        except Exception as e:
            print(f"[error] vision unavailable: {e}")
            return 1

    sleep_words = ([w for w in args.sleep_word.split(",")] if args.sleep_word
                   else (m.sleep_words if m.sleep_words else None))
    loop = VoiceLoop(stt, chat.reply, app, barge_in=args.barge_in, wake_words=wake_words,
                     idle_timeout=args.idle_timeout, start_engaged=not args.start_dormant,
                     sleep_words=sleep_words, barge_in_ms=args.barge_in_ms, barge_in_boost=args.barge_in_boost)
    loop.vision = watcher
    loop.add_context = chat.add_context      # a visual question hands her the current scene
    loop.dictation_pause_s = args.dictation_pause
    loop.dictation_words = list(m.dictation_words)
    loop.dictation_end_words = list(m.dictation_end_words)
    if getattr(app, "pipeline", None) is not None:
        loop.on_dictation = lambda on: app.pipeline.speak("Listening.") if on else None
    note_threads: List[threading.Thread] = []
    if runner is not None or memory_on:
        prev_event = loop.on_event

        def remember_you(kind, text):
            if kind == "you":
                last_you["text"] = text
            prev_event(kind, text)
        loop.on_event = remember_you
    if runner is not None:
        runner.deliver = loop.announce       # a finished task is told when the room is quiet
        runner.start()
        loop.errands = runner
        print(f"[errands] on: {agent_url}, polled every {args.errand_poll:.0f}s; "
              f"{len(ledger.open_items())} open in tasks.md")
    if memory_on:
        def session_end(reason):
            def work():
                try:
                    text = chat.summarise(
                        "The conversation is ending. For your own notes, in at most three plain sentences: "
                        "what happened, any decision or fact worth remembering, and anything left to follow "
                        "up. No greetings, no tags, no markdown.")
                except Exception as e:
                    print(f"[memory] session summary failed: {e}")
                    return
                if text:
                    notebook.session_summary(text)
                    print(f"[memory] session summary written ({reason})")
            t = threading.Thread(target=work, daemon=True, name="session-note")
            note_threads.append(t)
            t.start()
        loop.on_session_end = session_end
        print(f"[memory] on: {notebook.path} ({len(notebook.recent(100000).split(chr(10)))} lines), "
              f"{ledger.path} ({len(ledger.items)} tasks)")
    print(f"[log] this session is being written to {log_path}")
    if wake_words:
        names = ", ".join(repr(w) for w in loop.wake_words)
        print(f"[mode] {'dormant, listening for ' + names if not loop.engaged else 'engaged; after ' + str(int(args.idle_timeout)) + 's of quiet, wakes on ' + names}")
    app.on_submit = loop.on_user_text        # typed text goes through Claude too

    def toggle_waiting(on=None):
        loop.set_waiting((not loop.waiting) if on is None else on, "spacebar" if on is None else "panel")
        app.waiting = loop.waiting
    app.on_wait_toggle = toggle_waiting
    app.on_dictation_toggle = lambda: loop.set_dictation(not loop.dictating, "Tab")

    # ambience: files in sounds/idle/ play at random while nothing is happening
    idle = None
    if face_dir:
        from .idle_sounds import IdleSounds
        idle_bank = SoundBank(os.path.join(face_dir, m.sounds, "idle"))
        if not idle_bank.names:                  # no idle/ subfolder: the sound effects double as ambience
            idle_bank = sounds
        if idle_bank.names:
            cfg = m.idle_sounds or {}
            pipeline_ = getattr(app, "pipeline", None)
            idle = IdleSounds(idle_bank,
                              is_quiet=lambda: not loop.waiting and not (pipeline_ is not None and pipeline_.is_busy) and not loop._thinking
                              and not bool(getattr(stt, "speech_active", False)) and time.monotonic() - loop._last_busy > 3,
                              interval=cfg.get("interval", (30, 90)), quiet_for=cfg.get("quiet_for", 10))
            idle.start()
            loop.on_reply_start = idle.hush          # fade the ambience the moment a reply begins
            print(f"[idle] sounds every {idle.interval[0]:.0f}-{idle.interval[1]:.0f} s when quiet: {', '.join(idle_bank.names)}")

    panel = None
    if not args.no_web:
        panel = _start_panel(args, p, loop, app, audio, stt, chat, backend, watcher, m, face_dir, idle,
                             runner=runner, notebook=notebook if memory_on else None)

    if stt is not None:
        try:
            audio.start_mic(on_frames=loop.process, rate=stt.sample_rate,
                            device=args.mic_device, open_rate=args.mic_rate)
        except Exception as e:
            print(f"[error] mic unavailable: {e}")
            return 1
        print(f"[voice] listening ({stt.name}) — talk to the face. Esc quits."
              + (f"  voice={voice}" if voice else "")
              + (f"  persona={len(character.split())} words" if character else ""))
    else:
        print("[voice] text-only: press Enter in the window, type, Enter to send to Claude.")

    try:
        app.run()
    finally:
        loop.end_session("window closed")        # the memory's session note, if there was a session
        if watcher is not None:
            watcher.stop()
        if runner is not None:
            runner.stop()
        if panel is not None:
            panel.stop()
        if idle is not None:
            idle.stop()
        for t in note_threads:                   # give the summary up to 10 s, then leave anyway
            t.join(timeout=10)
    return 0


def _start_panel(args, parser, loop, app, audio, stt, chat, backend, watcher, manifest, face_dir, idle=None,
                 runner=None, notebook=None):
    """The control page: registers what it may read and change, then serves it."""
    from .web_panel import WebPanel
    panel = WebPanel(port=args.web_port, host=args.web_host)
    panel.parser, panel.args, panel.manifest = parser, args, manifest
    pipeline = getattr(app, "pipeline", None)
    last = {"heard": "", "said": "", "turn": ""}

    orig_event = loop.on_event

    def on_event(kind, text):
        orig_event(kind, text)
        panel.record(kind, text)
        if kind == "hearing":
            last["heard"] = text
        elif kind == "bot":
            last["said"] = text
        elif kind == "turn":
            last["turn"] = text
    loop.on_event = on_event

    def status():
        st = {"face": manifest.name, "waiting": loop.waiting,
              "stt": getattr(stt, "name", "text only") if stt else "text only",
              "llm": f"{args.llm} {chat.model}", "tts": f"{backend.name} {getattr(backend, 'voice_name', '') or ''}".strip(),
              "mic_rms": audio.get_state()[0], "speaking": bool(pipeline and pipeline.is_busy),
              "thinking": bool(getattr(loop, "_thinking", False)), "engaged": loop.engaged,
              "first_audio_ms": (pipeline.stats or {}).get("time_to_first_audio_ms") if pipeline else None,
              "last_heard": last["heard"], "last_said": last["said"], "last_turn": last["turn"]}
        if watcher is not None:
            n = watcher.latest()
            st["vision"] = {"described": watcher.stats.get("described", 0),
                            "skipped": watcher.stats.get("skipped_unchanged", 0),
                            "latest": (n.changes or n.notes) if n else ""}
        st["dictation"] = loop.dictating
        if runner is not None:
            st["errands"] = {"backend": runner.url, "reachable": runner.reachable,
                             "open": len(runner.pending()), "done": runner.stats["done"],
                             "failed": runner.stats["failed"], "latest": runner.last_summary}
        return st
    panel.status_fn = status

    if stt is not None and hasattr(stt, "set_silence_ms"):
        panel.tunable("silence_ms", lambda: int(round(stt.endpoint_delay_s * 1000)), stt.set_silence_ms,
                      "Pause that ends your turn. Lower is snappier; below ~400 it starts cutting pauses mid-sentence.",
                      kind="int", unit="ms", lo=150, hi=2000, flag="--silence-ms")
    panel.tunable("barge_in", lambda: loop.barge_in, lambda v: setattr(loop, "barge_in", v),
                  "Keep listening while the character talks and interrupt it when someone speaks. Needs a mic that cannot hear the speaker.",
                  kind="bool", flag="--barge-in")
    panel.tunable("barge_in_ms", lambda: loop.barge_in_ms, lambda v: setattr(loop, "barge_in_ms", v),
                  "Continuous speech needed before a barge-in counts.", kind="int", unit="ms", lo=100, hi=2000, flag="--barge-in-ms")
    panel.tunable("barge_in_boost", lambda: loop.barge_in_boost, lambda v: setattr(loop, "barge_in_boost", v),
                  "How much louder than the usual onset threshold speech must be while the character talks. Raise if it interrupts itself.",
                  kind="float", lo=1.0, hi=10.0, flag="--barge-in-boost")
    panel.tunable("barge_in_duty", lambda: loop.barge_in_duty, lambda v: setattr(loop, "barge_in_duty", v),
                  "Share of the barge-in window the mic must be loud. A voice is about 0.5, a knock about 0.2. Raise if bangs get through, lower if your voice does not.",
                  kind="float", lo=0.1, hi=1.0)
    panel.tunable("mic_highpass", lambda: audio.mic_high_pass,
                  lambda v: setattr(audio, "mic_high_pass", float(v)),
                  "High-pass on the microphone, in Hz. Cuts hum, air conditioning and desk thumps "
                  "before the speech gate sees them. Raise it in a noisy room; 0 turns it off.",
                  kind="float", unit="Hz", lo=0, hi=400, flag="--mic-highpass")
    panel.tunable("mic_lowpass", lambda: audio.mic_low_pass,
                  lambda v: setattr(audio, "mic_low_pass", float(v)),
                  "Low-pass before the microphone is resampled down, in Hz. Anti-aliasing; only "
                  "does anything when the device runs faster than the recognizer. 0 turns it off.",
                  kind="float", unit="Hz", lo=0, hi=20000, flag="--mic-lowpass")
    panel.tunable("echo_threshold", lambda: loop.echo_threshold, lambda v: setattr(loop, "echo_threshold", v),
                  "Mic/speaker loudness correlation above this is treated as the character's own voice, not a barge-in.",
                  kind="float", lo=0.0, hi=1.0)
    panel.tunable("idle_timeout", lambda: loop.idle_timeout, lambda v: setattr(loop, "idle_timeout", v),
                  "Wake mode: seconds of quiet after its own last reply before it goes dormant.", kind="float", unit="s",
                  lo=5, hi=3600, flag="--idle-timeout")
    panel.tunable("waiting", lambda: loop.waiting, lambda v: app.on_wait_toggle and (loop.set_waiting(v, "panel"), setattr(app, "waiting", loop.waiting)),
                  "Waiting mode (the spacebar in the window does the same): stops talking, ignores the mic, wake words and typed text until switched off. For calls and meetings.",
                  kind="bool")
    panel.tunable("engaged", lambda: loop.engaged,
                  lambda v: loop.engage("panel") if v else loop.disengage("panel"),
                  "Wake mode: on = answering; off = dormant until a wake word.", kind="bool")
    panel.tunable("dictation", lambda: loop.dictating, lambda v: loop.set_dictation(v, "panel"),
                  "Dictation mode (Tab in the window does the same): keeps transcribing but only answers "
                  "after dictation_pause_s of quiet, so you can dictate a paragraph or think aloud. "
                  "Switching it off answers what was dictated.", kind="bool")
    panel.tunable("dictation_pause_s", lambda: loop.dictation_pause_s,
                  lambda v: setattr(loop, "dictation_pause_s", float(v)),
                  "Dictation mode: seconds of quiet before everything dictated is answered as one turn.",
                  kind="float", unit="s", lo=1, hi=60, flag="--dictation-pause")
    if runner is not None:
        panel.tunable("errand_poll_s", lambda: runner.poll_s, lambda v: setattr(runner, "poll_s", float(v)),
                      "Seconds between polls of the backend agent for finished tasks.",
                      kind="float", unit="s", lo=1, hi=120, flag="--errand-poll")
    if idle is not None:
        def set_gap(lo=None, hi=None):
            a, b = idle.interval
            a, b = (float(lo) if lo is not None else a), (float(hi) if hi is not None else b)
            idle.interval = (min(a, b), max(a, b))
        panel.tunable("idle_gap_min", lambda: idle.interval[0], lambda v: set_gap(lo=v),
                      "Ambient sounds: shortest wait between two, in seconds.", kind="float", unit="s", lo=2, hi=3600)
        panel.tunable("idle_gap_max", lambda: idle.interval[1], lambda v: set_gap(hi=v),
                      "Ambient sounds: longest wait between two, in seconds.", kind="float", unit="s", lo=2, hi=3600)
        panel.tunable("idle_quiet_for", lambda: idle.quiet_for, lambda v: setattr(idle, "quiet_for", float(v)),
                      "Ambient sounds: how long the room must have been quiet before one plays.", kind="float", unit="s", lo=0, hi=600)
    if getattr(app, "schedule", None) is not None:
        panel.tunable("emotion_hold", lambda: app.schedule.emotion_hold,
                      lambda v: setattr(app.schedule, "emotion_hold", float(v)),
                      "Seconds an expression holds after its tag. If no new tag arrives the face settles "
                      "back to neutral. 0 keeps the last mood indefinitely.",
                      kind="float", unit="s", lo=0, hi=3600)
    panel.tunable("sfx_over_speech", lambda: manifest.sfx_over_speech,
                  lambda v: None, "Whether a {{sfx}} sound plays over her voice at its word, or waits "
                  "until she stops talking. Set in face.json; a restart applies a change.", kind="bool")
    panel.tunable("debug_overlay", lambda: app.debug, lambda v: setattr(app, "debug", v),
                  "Viseme, emotion, fps and timing overlay on the face window.", kind="bool", flag="--debug")
    if watcher is not None:
        panel.tunable("vision_interval", lambda: watcher.interval, lambda v: setattr(watcher, "interval", v),
                      "Seconds between camera bursts.", kind="float", unit="s", lo=2, hi=120, flag="--vision-interval")
        panel.tunable("vision_change", lambda: watcher.change_threshold, lambda v: setattr(watcher, "change_threshold", v),
                      "Mean pixel change needed before a burst is sent to the vision model (0.035 = 3.5%).",
                      kind="float", lo=0.0, hi=0.5, flag="--vision-change")
        src = getattr(watcher, "source", None)
        if src is not None and hasattr(src, "burst"):
            def snapshot():
                frames = src.burst(1)
                return frames[0] if frames else None
            panel.snapshot = snapshot

    if pipeline is not None:
        panel.action("speak", lambda t: (pipeline.speak(t or "Testing one two three. Can you hear me from the door?"), "speaking")[1],
                     "Say this through the speaker with the face (a [tag] works). Empty = the test phrase.", takes_text=True)
        panel.action("interrupt", lambda t: (pipeline.interrupt(), "stopped")[1], "Stop speaking now.")
    panel.action("say as visitor", lambda t: (loop.on_user_text(t), "sent")[1] if t.strip() else "type something first",
                 "Send this line to the brain as if a visitor said it.", takes_text=True)
    panel.action("go dormant", lambda t: (loop.disengage("panel"), "dormant")[1], "Wake mode: stop answering until a wake word.")
    if runner is not None:
        panel.action("task", lambda t: f"queued {runner.submit(t).id}" if t.strip() else "type the task first",
                     "Hand this to the backend agent now, as if the character had written {{task ...}}. "
                     "The result is announced when the room is quiet.", takes_text=True)
    if notebook is not None:
        panel.action("note", lambda t: ("noted: " + notebook.note(t)) if t.strip() else "type the note first",
                     "Append a line to the character's notes.md.", takes_text=True)

    if pipeline is not None and stt is not None:
        from .calibrate import run_calibration
        panel.calibrator = lambda ask: run_calibration(audio, pipeline.speak, lambda: pipeline.is_busy,
                                                       args.mic_device, args.output_device, ask=ask)
        panel.pause = lambda on: setattr(loop, "paused", on)
    panel.start()
    return panel


if __name__ == "__main__":
    sys.exit(main())
