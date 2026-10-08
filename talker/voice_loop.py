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

from .actions import strip_actions
from .phoneme_scheduler import strip_tags
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
        self.echo_hold = 1.5           # an echo verdict stands this long: the correlation wobbles as she talks
        self._echo_last = (0.0, 0.0)   # (time, corr) of the last echo verdict
        self._spoke_at = 0.0           # last time she was heard speaking (a barge-in does not clear it)
        self._echo_started: Optional[float] = None   # when this reply's mic history began
        self.on_event = on_event or (lambda kind, text: print(f"[{kind}] {text}"))
        self.clock = clock
        # Wake mode: with wake words set, nothing is answered until one is heard;
        # after `idle_timeout` s of silence, or when the brain ends the chat, go dormant again.
        # longest phrases first so "hey eve" wins over "eve"
        self.wake_words = sorted({w.strip().lower() for w in (wake_words or []) if w.strip()},
                                 key=lambda w: (-len(w.split()), -len(w)))
        self.idle_timeout = idle_timeout
        # News (a finished errand) wakes a dormant character to say it, then the usual idle
        # timeout applies again. Off: the news waits for the next wake word.
        self.announce_wakes = True
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
        self._last_said = ""           # her last reply, to recognise her own voice coming back in
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
        """Called from the audio callback: hand off and return immediately. The frame is
        stamped here, not when the recognizer gets to it: Whisper can hold that thread for
        a second, and the echo guard compares this clock with the speaker's."""
        try:
            self._audio_q.put_nowait((time.monotonic(), pcm))
        except queue.Full:
            pass                      # recognizer is far behind; drop rather than block the mic

    def _drain(self) -> None:
        while True:
            t_in, pcm = self._audio_q.get()
            try:
                self._process(pcm, t_in)
            except Exception as e:
                self.on_event("error", f"STT failed: {e}")

    def _process(self, pcm: bytes, t_in: Optional[float] = None) -> None:
        t_in = time.monotonic() if t_in is None else t_in     # when the mic heard this frame
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
            self._spoke_at = time.monotonic()
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
            if self._echo_started is None:            # a new reply: start the echo history afresh
                self._echo_started = t_in
                self._echo.reset()
            self._echo.add_mic(t_in, float(_np.sqrt(_np.mean(frame * frame))) if frame.size else 0.0)
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
                    if env and not self._echo.ready(t_in):
                        return                            # too early in her reply to tell echo from a person
                    corr = self._echo.correlation(env(), t_in) if env else 0.0
                    t_e, c_e = self._echo_last
                    held = t_in - t_e < self.echo_hold and c_e >= self.echo_threshold
                    if corr >= self.echo_threshold or held:
                        if corr >= self.echo_threshold:
                            self._echo_last = (t_in, corr)
                        self.on_event("echo", f"mic follows the speaker (corr {corr:.2f}"
                                      + (f", {c_e:.2f} a moment ago" if held and corr < self.echo_threshold else "")
                                      + "); not a barge-in")
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
        self._echo_started = None
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
        if loud / len(self._loud) < self.barge_in_duty:
            return False
        # A clap or a knock is loud at the start of the window and rings down; speech is
        # still loud when the window closes. Require a loud frame in the last third.
        t_end = self._loud[-1][0]
        span = t_end - self._loud[0][0]
        if span <= 0:
            return True
        return any(r >= thr for t, r in self._loud if t >= t_end - span / 3)

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

    def _strip_echo(self, text: str) -> str:
        """With barge-in, the first words the mic hears are often her own sentence coming
        back through the speaker, followed by what the person said. Drop a leading run
        of words that matches her last reply; drop the lot if it was all her."""
        said = self._norm(strip_tags(strip_actions(self._last_said)))
        raw = text.split()
        words = self._norm(text)
        if len(said) < 3 or len(words) < 3 or len(raw) != len(words):
            return text
        import difflib
        sm = difflib.SequenceMatcher(None, words, said, autojunk=False)
        blocks = sorted((a, n) for a, _b, n in sm.get_matching_blocks() if n)
        # Hers: any run of three or more of her words, wherever it sits (the mic can stitch
        # her sentence and the person's together), plus a lone mis-heard word between two runs.
        hers = [False] * len(words)
        for a, n in blocks:
            if n >= 3 or (a == 0 and n >= 2):
                for i in range(a, a + n):
                    hers[i] = True
        for i in range(1, len(words) - 1):
            if not hers[i] and hers[i - 1] and hers[i + 1]:
                hers[i] = True
        if not any(hers):
            return text
        kept = [w for w, h in zip(raw, hers) if not h]
        matched = sum(n for _a, n in blocks)
        if len(kept) < 2 or matched >= 0.8 * len(words):
            self.on_event("echo", f"her own voice, dropped: {text}")
            return ""
        self.on_event("echo", "dropped her own words: " + " ".join(w for w, h in zip(raw, hers) if h))
        return " ".join(kept)

    def on_user_text(self, text: str) -> None:
        """Handle a finished user utterance (also used by --text-only)."""
        if self.waiting:
            self.on_event("ignored", f"waiting mode: {text}")
            return
        # Only with barge-in does the mic listen while she talks. Half-duplex is deaf
        # then, so an utterance is always the person, and one that answers her by
        # repeating her own words ("...somewhere you can speak freely?" "Yes, I can
        # speak freely.") must not be thrown away as her echo.
        if self.barge_in and self._last_said and time.monotonic() - self._spoke_at < 30:
            text = self._strip_echo(text)
            if not text.strip():
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
    PARTIAL_STALE_S = 4.0          # a partial with no newer words for this long is not 'mid-sentence'

    def announce(self, event_text: str) -> bool:
        """Tell the brain something happened and let it speak up, but only when the room is
        quiet: not while it talks or thinks, not while someone is mid-sentence, not within
        a couple of seconds of either. Returns False to say: try again later."""
        if self.waiting or self.paused or self.dictating:
            return False
        if not self.engaged and not (self.announce_wakes and self.wake_words):
            return False
        if self.speaker.is_busy or self._thinking:
            return False
        now = time.monotonic()
        # Someone mid-sentence, unless that partial went stale: a cloud recognizer can send a
        # partial and never its final, which would otherwise hold the news back for good.
        if self._partial and now - self._last_heard < self.PARTIAL_STALE_S:
            return False
        if now - self._last_heard < 2.0 or self.clock() - self._last_busy < 2.0:
            return False
        with self._lock:
            if self._thinking:
                return False
            self._thinking = True
        if not self.engaged:
            self.engage("news to tell")               # stays engaged for idle_timeout after, for follow-ups
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

    def brain_health(self, ok: bool, kind: str = "", detail: str = "") -> None:
        """Track whether the brain answers; a change of state is logged and recorded once."""
        was = getattr(self, "brain", None) or {"ok": True}
        if ok and not was.get("ok", True):
            print(f"[brain] answering again (was: {was.get('kind')})")
            self.on_brain(True, "", "")
        elif not ok and (was.get("ok", True) or was.get("kind") != kind):
            print(f"[brain] failing: {kind}")
            self.on_brain(False, kind, detail)
        self.brain = {"ok": ok, "kind": "" if ok else kind, "since": was.get("since") if ok == was.get("ok", True)
                      else time.strftime("%H:%M:%S")}

    def on_brain(self, ok: bool, kind: str, detail: str) -> None:
        """Hook: replaced in main() to record incidents."""

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
                # A reply that was nothing but {{blocks}} is silence to the listener: give it words.
                from .actions import strip_actions
                from .phoneme_scheduler import strip_tags
                if reply and not strip_tags(strip_actions("".join(reply))).strip():
                    yield " Let me check."
            except Exception as e:
                self.on_event("error", f"LLM failed: {e}")
                kind, spoken = brain_problem(e)
                self.brain_health(False, kind, str(e))
                yield spoken
            else:
                self.brain_health(True)
            finally:
                self._thinking = False
                self._last_activity = self.clock()
                self._last_said = "".join(reply)
                self.on_event("bot", "".join(reply).strip())
                if self._ended:
                    self.disengage("the conversation ended")


        self.speaker.speak_stream(timed_chunks())


# ═══════════════════════════════════════════════════════
# PROFILES
# ═══════════════════════════════════════════════════════
def brain_problem(e: Exception) -> tuple:
    """(kind, what to say) for a failed brain call, so the person hears why, not just 'sorry'."""
    msg = f"{type(e).__name__}: {e}".lower()
    if "credit balance" in msg or "billing" in msg:
        return "out_of_credit", "[sad]I can't think right now: my thinking service is out of credit."
    if "401" in msg or "authentication" in msg or "api key" in msg or "x-api-key" in msg:
        return "bad_key", "[sad]I can't think right now: my thinking service won't accept my key."
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return "rate_limited", "[sad]I'm being rate limited for a moment. Ask me again shortly."
    if "529" in msg or "overloaded" in msg or " 50" in msg or "internal server" in msg:
        return "overloaded", "[sad]My thinking service is overloaded right now. Try me again in a minute."
    if "connect" in msg or "timeout" in msg or "timed out" in msg or "resolution" in msg:
        return "unreachable", "[sad]I can't reach my thinking service right now."
    return "error", "[sad]Sorry, I could not think of an answer just now."


def with_clock(reply, add_context, every_s: float = 600.0, clock=None):
    """The date and time in the system prompt are frozen at launch (rewriting the prompt each
    turn would throw its cache away). Before a turn, when `every_s` has passed since the last
    time note or the date has changed, hand the brain the current date, time and place as a
    context note. A few tokens, now and then; works with every brain."""
    from datetime import datetime
    from .launch_facts import now_text
    clock = clock or (lambda: datetime.now().astimezone())
    last = {"at": None}

    def wrapped(text, *a, **k):
        now = clock()
        prev = last["at"]
        if prev is None or (now - prev).total_seconds() >= every_s or now.date() != prev.date():
            add_context(now_text(now))
            last["at"] = now
        return reply(text, *a, **k)
    return wrapped


HANDOFF_ACK = {"task": "On it.", "approve": "Okay, going ahead.", "deny": "Alright, I won't."}


def handoff_brief(reply):
    """A reply that STARTS with {{task}}, {{approve}} or {{deny}} is a handoff: say a short
    acknowledgement at once (before the block has even finished streaming), pass the block
    through for the action dispatcher, and drop any words after it, so the plan is never read
    back. Any other reply streams through untouched, with no added delay."""
    import re as _re
    lead = _re.compile(r"^\s*(?:\[[^\]\n]{0,40}\]\s*)*")
    kinds = "|".join(HANDOFF_ACK)

    def wrapped(text, *a, **k):
        buf, mode, pending, in_block = "", "decide", "", True
        for chunk in reply(text, *a, **k):
            if mode == "pass":
                yield chunk
                continue
            if mode == "decide":
                buf += chunk
                rest = buf[lead.match(buf).end():]
                if not rest or (rest.startswith("{") and _re.fullmatch(r"\{\{?\s*[a-z]*", rest)):
                    continue                              # can't tell yet: a block may be starting
                m = _re.match(r"\{\{\s*(" + kinds + r")\b", rest)
                if not m:
                    mode = "pass"
                    yield buf
                    continue
                mode = "handoff"
                yield buf[:len(buf) - len(rest)] + HANDOFF_ACK[m.group(1)] + " "
                chunk = rest
            pending += chunk
            while pending:                                # blocks go through; words between them don't
                if in_block:
                    end = pending.find("}}")
                    if end < 0:
                        keep = 1 if pending.endswith("}") else 0
                        out, pending = pending[:len(pending) - keep], pending[len(pending) - keep:]
                        if out:
                            yield out
                        break
                    yield pending[:end + 2]
                    pending, in_block = pending[end + 2:], False
                else:
                    start = pending.find("{{")
                    if start < 0:
                        pending = "{" if pending.endswith("{") else ""
                        break
                    pending, in_block = pending[start:], True
        if mode == "decide" and buf:
            yield buf                                     # a reply too short to decide on: say it as it is
    return wrapped


def always_speaks(reply):
    """A reply made only of {{blocks}} and [tags] is silence to the listener: add a word."""
    import re as _re

    def wrapped(text, *a, **k):
        said = []
        for chunk in reply(text, *a, **k):
            said.append(chunk)
            yield chunk
        full = "".join(said)
        spoken = _re.sub(r"\{\{.*?\}\}|\[[^\]\n]{1,40}\]", "", full, flags=_re.S)
        if full.strip() and not _re.search(r"[A-Za-z0-9]", spoken):
            yield " Noted."
    return wrapped


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
    if args.stt == "elevenlabs":
        return {"silence_ms": args.silence_ms, "language": args.stt_language or None}
    if args.stt == "openai":
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
    p.add_argument("--stt-language", default="en",
                   help="Language code for cloud Scribe (--stt elevenlabs); \"\" lets it auto-detect, which "
                        "drifts into other languages on short or hesitant English. Whisper and OpenAI are always en")
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
                   help="Model id for the chosen --llm; overrides the face's own \"models\" entry "
                        "(defaults: claude-haiku-4-5 for speed; "
                        "--model claude-opus-5 for the best writing at ~2 s more per reply; openai: gpt-4o-mini). "
                        "Add -fast to an Opus model (claude-opus-5-fast, claude-opus-4-8-fast) for fast mode: "
                        "the same model at up to 2.5x the tokens per second and twice the price. It does not "
                        "shorten the wait for the first token, only the words after it, so it reaches the first "
                        "spoken sentence sooner. Research preview: it falls back to standard speed if your key "
                        "has no access")
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
    p.add_argument("--vision-model", default="claude-haiku-4-5", help="Claude vision model for scene notes")
    p.add_argument("--vision-mode", choices=["on_demand", "periodic"], default="on_demand",
                   help="on_demand: look at startup, on waking from dormant, and when she decides to ({{look}} or a "
                        "visual question); periodic: every --vision-interval seconds while awake")
    p.add_argument("--vision-backend", choices=["auto", "ollama", "anthropic"], default="anthropic",
                   help="Who describes the camera: ollama (a local vision model, free per look), anthropic "
                        "(Claude; ANTHROPIC_VISION_API_KEY gives it its own key), auto = local first, Claude fallback")
    p.add_argument("--vision-url", default=None,
                   help="Ollama server for --vision-backend ollama/auto (default VISION_OLLAMA_URL, else http://localhost:11434)")
    p.add_argument("--vision-local-model", default=None,
                   help="Local vision model (default VISION_OLLAMA_MODEL, else qwen3.8:27b)")
    p.add_argument("--vision-dormant-interval", type=float, default=300.0,
                   help="Seconds between camera looks while dormant (default 300); sound still triggers a look, "
                        "at most once a minute")
    p.add_argument("--clock-every-min", type=float, default=10.0,
                   help="Remind the brain of the current date, time and place before a turn when this many "
                        "minutes have passed since the last reminder, or the date changed (default 10)")
    p.add_argument("--dream-model", default="claude-opus-5-5",
                   help="Model for the nightly self-review (dream.py): claude-opus-5-5, or claude-fable-5-1 for the deepest")
    p.add_argument("--dream-idle-hours", type=float, default=3.0,
                   help="Dream once a day after this many hours dormant with no conversation (default 3)")
    p.add_argument("--no-dream", action="store_true", help="No nightly self-review")
    p.add_argument("--planner-model", default="claude-sonnet-5",
                   help="Model that plans agent goals into steps (default claude-sonnet-5)")
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
                        "with your agent harness), e.g. http://agentbox:8030; default: what the control page "
                        "saved in settings.json, then AGENT_RELAY_URL from .env. Only a face with \"errands\" uses it")
    p.add_argument("--errand-mode", choices=["auto", "ask"], default="auto",
                   help="auto: she plans, chases and approves what the request itself asked for, and only asks "
                        "the person for real decisions; ask: every action the agent wants to take is put to the person")
    p.add_argument("--errand-poll", type=float, default=5.0,
                   help="Seconds between polls of the backend agent for finished tasks (default 5)")
    p.add_argument("--agent-timeout", type=float, default=10.0,
                   help="How long each request to the backend agent may take before it counts as "
                        "unreachable (default 10 s). Raise it when the agent is across a VPN or a "
                        "slow network: on a LAN nothing waits this long anyway")
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
    # Slow imports run in the background while the face and devices load: anthropic takes
    # ~15 s to import on a Pi Zero. A later `import` waits for it rather than doing it twice.
    threading.Thread(target=lambda: __import__("anthropic"), name="preload", daemon=True).start()
    from .phoneme_scheduler import warm_up_g2p_in_background
    warm_up_g2p_in_background()
    # The control page comes up first and says what is loading, so a slow start does not look
    # like a dead one; its settings appear once everything is up.
    starting = {"stage": "loading libraries", "since": time.time()}
    early_panel = None
    if not args.no_web:
        from .web_panel import WebPanel
        early_panel = WebPanel(port=args.web_port, host=args.web_host)
        early_panel.status_fn = lambda: {"face": args.face, "starting": starting["stage"],
                                         "starting_s": round(time.time() - starting["since"])}
        early_panel.start()

    def stage(text):
        starting["stage"] = text
        print(f"[start] {text}")
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
    stage("loading the face")
    assets = FaceAssetLoader().load(face_dir) if face_dir else FaceAssetLoader().build(default_manifest(args.face))

    # Face-level defaults for voice, model and persona (flags win)
    m = assets.manifest
    args.tts = (args.tts or m.tts or            # face.json "tts" picks this face's own backend
                ("elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "edge")).lower()
    voice = args.voice or m.voices.get(args.tts)
    tts_model = args.tts_model or m.tts_model or ("eleven_v3" if args.tts == "elevenlabs" else None)
    character = args.character or m.character or None
    model = args.model or m.models.get(args.llm)      # face.json "models", keyed by brain
    speed = args.voice_speed or m.voice_speed

    # Actions the brain may write next to its speech ({{move nod}}, {{sfx creak}},
    # {{tool ...}}); it is only told about the ones this face has.
    from .actions import ActionDispatcher, SoundBank, ToolBox, action_rules
    from .body import NullBody
    sounds = SoundBank(os.path.join(face_dir, m.sounds) if face_dir else None)
    body = NullBody((m.body or {}).get("moves", []))
    tools = ToolBox()
    from .launch_facts import now_text
    tools.add("clock", "the current date, time, time zone and location", lambda _a: now_text())
    # An assistant: notes she writes herself, and errands for a backend agent (errands.py).
    from .brains.claude_chat import assistant_rules
    memory_on = bool(m.memory) and notebook is not None
    # The agent's address: the flag, then what the control page saved, then .env.
    from . import local_settings
    agent_url, agent_source = "", "none"
    for source, value in (("flag", args.agent_url), ("settings", local_settings.load().get("agent_url")),
                          ("env", os.environ.get("AGENT_RELAY_URL"))):
        if value and str(value).strip():
            agent_url, agent_source = str(value).strip(), source
            break
    errands_on = bool(m.errands)
    if errands_on and not agent_url:
        print("[errands] face.json asks for a backend agent but no address is set yet: tasks wait until "
              "one is given on the control page (Agent tab), --agent-url or AGENT_RELAY_URL in .env")
    if errands_on:
        tools.add("tasks", "the task ledger with each task's state", lambda _a: ledger.render())
        print(f"[errands] can: {m.errands_can or '(not said; list what the agent can do under \"errands\" in face.json)'}")
    extra_rules = action_rules(body.moves, sounds.names, tools.describe()) + assistant_rules(memory_on, errands_on, m.errands_can)
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
            chat = ClaudeChat(model=model or "claude-haiku-4-5", effort=args.effort, character=character,
                              thinking=bool(args.thinking), can_see=can_see, wake_mode=bool(wake_words),
                              extra_rules=extra_rules)
        elif args.llm == "ollama":
            from .brains.ollama_chat import OllamaChat
            chat = OllamaChat(model=model, character=character, host=args.ollama_host,
                              can_see=can_see, wake_mode=bool(wake_words), extra_rules=extra_rules)
            print(f"[voice] loading {chat.model} on {chat.host} ...")
            chat.warm_up()                   # load the weights now, not on the first question
        else:
            from .brains.openai_compat_chat import OpenAICompatChat
            chat = OpenAICompatChat.openai(model=model, character=character, can_see=can_see,
                                           wake_mode=bool(wake_words), extra_rules=extra_rules)
    except Exception as e:                   # a wrong model name or a missing key, said plainly
        print(f"[error] brain '{args.llm}' unavailable: {e}")
        print("        python voice_loop.py --setup checks your keys and the models you have")
        return 1
    print(f"[voice] brain: {args.llm} {chat.model}"
          f"{' fast' if getattr(chat, 'speed', None) == 'fast' else ''}"
          f"{' (told it can see)' if can_see else ''}")
    stage("starting the voice and the display")
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
    orch = None
    last_you = {"text": ""}
    if errands_on:
        # She manages the agent work (orchestrator.py): a request becomes a planned goal whose
        # steps are sent in order, chased, retried and approved; one announcement per goal.
        from .errands import ErrandRunner
        from .orchestrator import Orchestrator, make_claude_planner
        ledger_state = {"planning": "in progress", "running": "in progress", "needs_input": "needs input",
                        "done": "done", "failed": "failed"}

        def on_goal_change(g):
            if g.state == "failed" and not getattr(g, "_incident", False):
                g._incident = True
                from . import incidents
                incidents.record("goal_failed", goal=g.text[:240], why=g.progress()[:400])
            if ledger.get(g.id) is None:
                ledger.add(g.id, g.text)
            ledger.set_state(g.id, ledger_state.get(g.state, "in progress"), summary=g.progress() or None)

        runner = ErrandRunner(agent_url, poll_s=args.errand_poll, sender=m.name,
                              timeout=args.agent_timeout)
        orch = Orchestrator(runner, planner=make_claude_planner(model=args.planner_model, can=m.errands_can),
                            announce=runner.say_later, mode=args.errand_mode, on_change=on_goal_change)
        runner.on_done, runner.on_fail = orch.on_errand_done, orch.on_errand_failed
        runner.on_needs_input = orch.on_errand_needs_input

        def task_handler(a):                       # returns at once; planning and HTTP run elsewhere
            task = f"{a.name} {a.args}".strip()
            if not task:
                return None
            g = orch.start_goal(task, context=f"the person had just said: {last_you['text']}" if last_you["text"] else "")
            print(f"[task] {g.id} goal: {g.text}")
            return None

        def decision_handler(approve):
            def handle(a):
                print(f"[task] {orch.decide(a.name.strip(), approve)}")
                return None
            return handle
        actions.register("task", task_handler)
        actions.register("approve", decision_handler(True))
        actions.register("deny", decision_handler(False))
    if getattr(app, "pipeline", None) is not None:
        app.pipeline.on_action = actions.dispatch

    stt = None
    if not text_only:
        try:
            from .stt_backends import make_stt
            stage("connecting speech recognition")
            stt = make_stt(args.stt, **stt_kwargs(args))
        except Exception as e:
            print(f"[error] STT backend '{args.stt}' unavailable: {e}")
            return 1

    watcher = None
    if args.camera is not None and not args.no_vision:
        stage("starting the camera")
    if args.camera is not None and not args.no_vision:
        try:
            from .vision import SceneWatcher, make_describer, open_camera
            source = open_camera(args.camera, tuning=local_settings.load().get("camera"))
            cam_index = source.label
            def on_note(n):
                if args.debug:
                    print(f"[scene] {'EMERGENCY ' if n.emergency else ''}people={n.people}: "
                          f"{n.changes or n.notes}")
                ctx = watcher.context()
                if ctx:
                    chat.add_context(ctx)                    # only fires on a real change

            describe = make_describer(args.vision_backend, claude_model=args.vision_model,
                                      ollama_url=args.vision_url or os.environ.get("VISION_OLLAMA_URL", "http://localhost:11434"),
                                      ollama_model=args.vision_local_model or os.environ.get("VISION_OLLAMA_MODEL", "qwen3.8:27b"))
            watcher = SceneWatcher(source, describe, dormant_interval=args.vision_dormant_interval,
                                   on_demand=args.vision_mode == "on_demand",
                                   interval=args.vision_interval, burst=args.vision_frames,
                                   change_threshold=args.vision_change, on_note=on_note)
            watcher.start()
            when = ("on demand (startup, waking up, {{look}}, visual questions)" if args.vision_mode == "on_demand"
                    else f"every {args.vision_interval:.0f}s, every {args.vision_dormant_interval:.0f}s while dormant")
            local = args.vision_local_model or os.environ.get("VISION_OLLAMA_MODEL", "qwen3.8:27b")
            print(f"[vision] on: camera {cam_index}, {args.vision_frames} frames {when}; {args.vision_backend} "
                  f"({local} / {args.vision_model}); frames are not stored (emergencies go to emergencies/)")
        except Exception as e:
            print(f"[error] vision unavailable: {e}")
            return 1

    sleep_words = ([w for w in args.sleep_word.split(",")] if args.sleep_word
                   else (m.sleep_words if m.sleep_words else None))
    brain = with_clock(chat.reply, chat.add_context, every_s=args.clock_every_min * 60)
    loop = VoiceLoop(stt, always_speaks(handoff_brief(brain)), app, barge_in=args.barge_in, wake_words=wake_words,
                     idle_timeout=args.idle_timeout, start_engaged=not args.start_dormant,
                     sleep_words=sleep_words, barge_in_ms=args.barge_in_ms, barge_in_boost=args.barge_in_boost)
    loop.vision = watcher
    if watcher is not None:
        watcher.is_dormant = lambda: bool(loop.wake_words) and not loop.engaged
        if watcher.on_demand:
            watcher.request(force=True)                 # one look at startup: the scene she starts from
            plain_engage = loop.engage

            def engage_and_look(reason=""):
                was = loop.engaged
                plain_engage(reason)
                if not was:
                    watcher.request(force=True)         # coming out of dormant: who is here now?
            loop.engage = engage_and_look

            def look_handler(a):
                def work():
                    seen = watcher.look_now(timeout=8.0)
                    if seen:
                        loop.announce(f"You looked: {seen}. Say what matters about it, briefly.")
                threading.Thread(target=work, daemon=True, name="look").start()
                return None
            actions.register("look", look_handler)

    def on_brain(ok, kind, detail):
        from . import incidents
        incidents.record("brain_ok" if ok else "brain_failed", kind=kind, detail=detail[:300])
    loop.on_brain = on_brain
    loop.add_context = chat.add_context      # a visual question hands her the current scene
    # Resilience: say when the cloud is gone (from a cached local recording: the voice is cloud
    # too), try to heal the link, and keep a record of what went wrong (logs/incidents.jsonl).
    from . import cues, incidents
    from .netwatch import NetWatch
    if not text_only:
        cues.prepare({"offline": "I've lost my internet connection. I'll keep trying, and tell you when I'm back.",
                      "online": "I'm back online."})

    def net_offline():
        incidents.record("net_offline")
        if not text_only:
            cues.play("offline")

    def net_online(seconds):
        incidents.record("net_online", down_s=round(seconds))
        if not text_only:
            cues.play("online")
    net = NetWatch(on_offline=net_offline, on_online=net_online)
    net.start()
    loop.net = net
    # Dream mode (dream.py): once a day, after a few idle hours, a strong model reviews the logs
    # and writes dreams/<date>.md with proposed improvements. Nothing is applied automatically.
    loop.dreams = None
    if not args.no_dream and not text_only:
        from .dream import DreamScheduler, dream as run_dream
        talk = {"at": time.time()}
        prev_talk_event = loop.on_event

        def note_talk(kind, text):
            if kind in ("you", "bot"):
                talk["at"] = time.time()
            prev_talk_event(kind, text)
        loop.on_event = note_talk
        extra = {"calibration.json": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration.json"),
                 "startup flags (clara-run)": os.path.expanduser("~/.local/bin/clara-run")}
        loop.dreams = DreamScheduler(
            run=lambda: run_dream(face_dir=face_dir or "", model=args.dream_model, extra_files=extra),
            is_dormant=lambda: bool(loop.wake_words) and not loop.engaged,
            last_activity=lambda: talk["at"], idle_hours=args.dream_idle_hours,
            online=lambda: net.online)
        loop.dreams.start()
        print(f"[dream] on: {args.dream_model}, once a day after {args.dream_idle_hours:g} h idle; reports in dreams/")
    prev_turn_event = loop.on_event

    def note_slow_turns(kind, text):
        if kind == "turn":
            try:
                ms = int(text.rsplit("first audio", 1)[1].split()[0])
                if ms > incidents.SLOW_TURN_MS:
                    incidents.record("slow_turn", first_audio_ms=ms, line=text)
            except (IndexError, ValueError):
                pass
        prev_turn_event(kind, text)
    loop.on_event = note_slow_turns
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
        for it in ledger.open_items():       # goals live in memory: say plainly that a restart cut them off
            ledger.set_state(it["id"], "failed", summary="interrupted: the character restarted before it finished")
        runner.start()
        loop.errands = runner
        print(f"[errands] on: {agent_url or '(no address yet)'}, polled every {args.errand_poll:.0f}s; "
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
                             runner=runner, notebook=notebook if memory_on else None, agent_source=agent_source,
                             orch=orch, panel=early_panel)
    print(f"[start] ready in {time.time() - starting['since']:.0f} s")

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
                 runner=None, notebook=None, agent_source="none", orch=None, panel=None):
    """The control page: registers what it may read and change, then serves it. `panel` is the
    one started early to show the startup stage; it is reused, not started twice."""
    from .web_panel import WebPanel
    early = panel is not None
    panel = panel or WebPanel(port=args.web_port, host=args.web_host)
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
              "llm": f"{args.llm} {chat.model}" + (" fast" if getattr(chat, "speed", None) == "fast" else ""), "tts": f"{backend.name} {getattr(backend, 'voice_name', '') or ''}".strip(),
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
        if getattr(loop, "dreams", None) is not None:
            st["dreaming"] = loop.dreams.dreaming
        st["brain"] = getattr(loop, "brain", None) or {"ok": True}
        if getattr(loop, "net", None) is not None:
            st["online"] = loop.net.online
            st["outages"] = loop.net.outages
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
    from . import local_settings as _ls
    panel.tunable("location", lambda: str(_ls.load().get("location") or ""),
                  lambda v: _ls.save("location", str(v).strip() or None),
                  "Where the character is, e.g. 'Bronxville, NY'. Used in time notes and {{tool clock}} at once, and "
                  "in the prompt from the next start. Empty = looked up from the internet address.", kind="str")
    panel.tunable("idle_timeout", lambda: loop.idle_timeout, lambda v: setattr(loop, "idle_timeout", v),
                  "Wake mode: seconds of quiet after its own last reply before it goes dormant.", kind="float", unit="s",
                  lo=5, hi=3600, flag="--idle-timeout")
    panel.tunable("waiting", lambda: loop.waiting, lambda v: app.on_wait_toggle and (loop.set_waiting(v, "panel"), setattr(app, "waiting", loop.waiting)),
                  "Waiting mode (the spacebar in the window does the same): stops talking, ignores the mic, wake words and typed text until switched off. For calls and meetings.",
                  kind="bool")
    panel.tunable("announce_wakes", lambda: loop.announce_wakes, lambda v: setattr(loop, "announce_wakes", v),
                  "Wake mode: a finished errand wakes the character to announce it, then it goes dormant "
                  "again after idle_timeout. Off: the news waits until someone says a wake word.", kind="bool")
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
        panel.tunable("agent_timeout_s", lambda: runner.timeout,
                      lambda v: setattr(runner, "timeout", float(v)),
                      "How long a request to the backend agent may take before it counts as "
                      "unreachable. Raise it over a VPN or a slow network.",
                      kind="float", unit="s", lo=2, hi=120, flag="--agent-timeout")
        from . import local_settings
        from .errands import AgentControl
        panel.agent = AgentControl(runner, can=manifest.errands_can, source=agent_source, orch=orch,
                                   save=lambda url: local_settings.save("agent_url", url or None))
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
        if src is not None and hasattr(src, "tune"):
            from . import local_settings

            def camera_setter(name):
                def apply(v):
                    try:
                        src.tune(name, v)
                    except ValueError as e:
                        print(f"[vision] {e}")
                        return
                    local_settings.save("camera", dict(src.settings))   # kept for the next start
                return apply
            cam_help = {
                "ev": "Exposure compensation in stops: + brightens (a room backlit by a window wants +0.5 to +1.5), "
                      "- darkens. Check with Test setup, Take a snapshot.",
                "metering": "What exposure is judged on: CentreWeighted (default), Spot (the middle only, best "
                            "against a bright window) or Matrix (the whole frame).",
                "brightness": "Brightness offset applied after exposure, -1 to 1 (0 = none).",
                "contrast": "Contrast, 0 to 4 (1 = normal). Raise a little in flat light.",
                "saturation": "Colour saturation, 0 to 4 (1 = normal).",
                "awb": "White balance: Auto, Daylight, Cloudy, Indoor, Incandescent, Tungsten, Fluorescent.",
            }
            for name in ("ev", "metering", "brightness", "contrast", "saturation", "awb"):
                numeric = src.NUMERIC.get(name)
                panel.tunable(f"camera_{name}", (lambda n=name: src.settings[n]), camera_setter(name),
                              cam_help[name], kind="float" if numeric else "str",
                              lo=numeric[1] if numeric else None, hi=numeric[2] if numeric else None,
                              options=None if numeric else list(src.MODES[name][2]))

    if pipeline is not None:
        panel.action("speak", lambda t: (pipeline.speak(t or "Testing one two three. Can you hear me from the door?"), "speaking")[1],
                     "Say this through the speaker with the face (a [tag] works). Empty = the test phrase.", takes_text=True)
        panel.action("interrupt", lambda t: (pipeline.interrupt(), "stopped")[1], "Stop speaking now.")
    panel.action("say as visitor", lambda t: (loop.on_user_text(t), "sent")[1] if t.strip() else "type something first",
                 "Send this line to the brain as if a visitor said it.", takes_text=True)
    if getattr(loop, "dreams", None) is not None:
        panel.action("dream now", lambda t: loop.dreams.dream_now(),
                     "Review the logs now and write a report with proposed improvements (Dreams tab). "
                     "Normally happens once a day after a few idle hours.")
        panel.dreams_dir = __import__("talker.dream", fromlist=["DREAMS"]).DREAMS
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
    if not early:
        panel.start()
    return panel


if __name__ == "__main__":
    sys.exit(main())
