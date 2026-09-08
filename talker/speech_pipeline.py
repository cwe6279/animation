"""
talker/speech_pipeline.py
==================
Glues a streaming TTS backend to the audio timeline and the viseme schedule.

    pipeline = SpeechPipeline(audio_engine, schedule_reader, backend)
    pipeline.start()
    pipeline.speak("[happy]Hello there! How are you?")     # whole string
    pipeline.speak_stream(llm_token_iterator)               # as tokens arrive
    pipeline.interrupt()                                    # barge-in
    pipeline.stop()

Each speak() call is one *session*: the text is split into sentences and
pushed to the backend as they become available, so the first sentence is
already playing while the rest is still being synthesized (or still being
written by an LLM). Sessions play back to back in the order they were
requested.

Timing model:
  * The backend reports word boundaries relative to its session's first
    audio sample.
  * The first PCM chunk we enqueue tells us where on the shared timeline the
    session actually starts (audio_engine.enqueue_pcm returns it).
  * Viseme/emotion events = session_start + backend time. If the device
    underruns mid-session (network stall) the session start is re-based from
    the next chunk so later words stay aligned.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from typing import Callable, Iterable, List, Optional, Tuple

from .actions import Action, parse_actions
from .audio_engine import BaseAudioEngine
from .phoneme_scheduler import (
    Emotion, EmotionEvent, ScheduleReader, SentenceSplitter,
    parse_tags, word_to_viseme_events, warm_up_g2p,
)
from .tts_backends import (
    AudioChunk, END_OF_TEXT, SentenceDone, TTSBackend, WordBoundary,
)

# Fire mouth shapes slightly before the sound: the renderer eases toward each
# target over ~50 ms, so leading by that much makes the peak land on the beat.
DEFAULT_LEAD_SECONDS = 0.04


class SpeechPipeline:
    def __init__(self, audio: BaseAudioEngine, schedule: ScheduleReader,
                 backend: TTSBackend, lead_seconds: float = DEFAULT_LEAD_SECONDS,
                 on_error: Optional[Callable[[str], None]] = None):
        self.audio = audio
        self.schedule = schedule
        self.backend = backend
        self.lead = lead_seconds
        self.on_error = on_error or (lambda msg: print(f"[speech] {msg}"))
        # {{move nod}} / {{sfx creak}} / {{tool ...}} blocks in the text: stripped with the
        # emotion tags and fired at the moment the words before them are spoken.
        self.on_action: Optional[Callable[[Action], None]] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._sessions: "asyncio.Queue[Optional[asyncio.Queue]]" = None  # type: ignore
        self._current: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._active_sessions = 0
        self.speech_end_time = 0.0       # timeline time when queued speech ends
        self.last_error: Optional[str] = None
        self.stats: dict = {}
        self.first_audio_at: float = 0.0   # time.monotonic() when the latest session's audio started
        self.first_sentence_at: float = 0.0  # when the first sentence was handed to the TTS backend

    # ── lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="speech-pipeline", daemon=True)
        self._thread.start()
        self._ready.wait()
        # Load g2p now (1-2 s once). Doing it in the background instead costs
        # ~600 ms on the first utterance through GIL contention.
        t0 = time.monotonic()
        warm_up_g2p()
        print(f"[speech] ready (g2p loaded in {(time.monotonic() - t0) * 1000:.0f} ms)")

    def stop(self) -> None:
        if self._loop is None:
            return
        self.interrupt()
        self._loop.call_soon_threadsafe(self._sessions.put_nowait, None)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._sessions = asyncio.Queue()
        self._ready.set()
        try:
            self._loop.run_until_complete(self._worker())
        finally:
            self._loop.close()

    async def _worker(self) -> None:
        try:
            await self.backend.warm_up()
        except Exception as e:
            self.on_error(f"backend warm-up failed: {e}")
        while True:
            sentences = await self._sessions.get()
            if sentences is None:
                try:
                    await self.backend.close()
                except Exception:
                    pass
                return
            self._current = asyncio.ensure_future(self._run_session(sentences))
            try:
                await self._current
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.last_error = str(e)
                self.on_error(f"TTS failed: {e}")
                traceback.print_exc()
            finally:
                self._current = None
                with self._lock:
                    self._active_sessions -= 1

    # ── public API ─────────────────────────────────────────────────
    @property
    def is_busy(self) -> bool:
        """True while any session is synthesizing or its audio is still queued."""
        with self._lock:
            active = self._active_sessions
        return active > 0 or self.audio.timeline_time() < self.speech_end_time

    def speak(self, text: str) -> None:
        self.speak_stream([text])

    def speak_stream(self, chunks: Iterable[str]) -> None:
        """
        Feed text incrementally (e.g. LLM tokens). Returns immediately; a
        helper thread drains `chunks` and hands sentences to the backend.
        """
        if self._loop is None:
            raise RuntimeError("SpeechPipeline.start() first")
        sentences: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._active_sessions += 1
        self._loop.call_soon_threadsafe(self._sessions.put_nowait, sentences)

        def feeder():
            splitter = SentenceSplitter()
            try:
                for chunk in chunks:
                    for s in splitter.feed(chunk):
                        self._loop.call_soon_threadsafe(sentences.put_nowait, s)
                for s in splitter.flush():
                    self._loop.call_soon_threadsafe(sentences.put_nowait, s)
            finally:
                self._loop.call_soon_threadsafe(sentences.put_nowait, END_OF_TEXT)

        threading.Thread(target=feeder, name="speech-feeder", daemon=True).start()

    def interrupt(self) -> None:
        """Stop speaking now: cancel synthesis, drop queued audio and schedule."""
        if self._loop is None:
            return

        def _cancel():
            # Drop sessions that haven't started.
            while not self._sessions.empty():
                try:
                    if self._sessions.get_nowait() is not None:
                        with self._lock:
                            self._active_sessions -= 1
                except asyncio.QueueEmpty:
                    break
            if self._current is not None and not self._current.done():
                self._current.cancel()

        self._loop.call_soon_threadsafe(_cancel)
        self.audio.flush()
        self.schedule.clear()
        self.speech_end_time = self.audio.timeline_time()

    # ── one session ────────────────────────────────────────────────
    async def _run_session(self, sentences: asyncio.Queue) -> None:
        t_request = time.monotonic()
        clean_sentences: asyncio.Queue = asyncio.Queue()
        tags: List[Tuple[int, Emotion]] = []      # (session word index, emotion)
        actions: List[Tuple[int, Action]] = []    # (session word index, action block)
        action_handles: List[asyncio.TimerHandle] = []
        words_in_text = 0
        pending_words: List[WordBoundary] = []
        session_start: Optional[float] = None
        session_frames = 0
        word_index = 0
        first_audio_at: Optional[float] = None
        text_done = False

        async def strip_tags():
            nonlocal words_in_text, text_done
            while True:
                s = await sentences.get()
                if s is END_OF_TEXT:
                    text_done = True
                    await clean_sentences.put(END_OF_TEXT)
                    return
                s, sentence_actions = parse_actions(s)
                for idx, act in sentence_actions:
                    actions.append((words_in_text + idx, act))
                clean, voiced, sentence_tags = parse_tags(s)
                for idx, emo in sentence_tags:
                    tags.append((words_in_text + idx, emo))
                if words_in_text == 0 and clean:
                    self.first_sentence_at = time.monotonic()
                words_in_text += len(clean.split())
                if clean:
                    await clean_sentences.put(voiced if self.backend.supports_audio_tags else clean)

        stripper = asyncio.ensure_future(strip_tags())

        def place_word(wb: WordBoundary) -> None:
            nonlocal word_index
            t0 = session_start + wb.start - self.lead
            t1 = session_start + wb.end - self.lead
            visemes = word_to_viseme_events(wb.word, t0, t1)
            emotions = [EmotionEvent(t0, emo) for idx, emo in tags if idx == word_index]
            self.schedule.append(visemes, emotions)
            for idx, act in actions:
                if idx == word_index:
                    action_handles.append(self._fire_at(t0 + self.lead, act))
            self.speech_end_time = max(self.speech_end_time, t1 + self.lead)
            word_index += 1

        try:
            async for ev in self.backend.synthesize(clean_sentences):
                if isinstance(ev, AudioChunk):
                    if not ev.pcm:
                        continue
                    chunk_start = self.audio.enqueue_pcm(ev.pcm, self.backend.sample_rate)
                    expected = None if session_start is None else session_start + session_frames / self.backend.sample_rate
                    if session_start is None:
                        session_start = chunk_start
                        first_audio_at = time.monotonic()
                        self.first_audio_at = first_audio_at
                        for wb in pending_words:
                            place_word(wb)
                        pending_words.clear()
                    elif chunk_start > expected + 0.005:
                        # Device underran (stall); re-base later words.
                        session_start += chunk_start - expected
                    session_frames += len(ev.pcm) // 2
                    self.speech_end_time = max(self.speech_end_time,
                                               chunk_start + len(ev.pcm) / 2 / self.backend.sample_rate)
                elif isinstance(ev, WordBoundary):
                    if session_start is None:
                        pending_words.append(ev)
                    else:
                        place_word(ev)
                elif isinstance(ev, SentenceDone):
                    pass
            # Tags positioned after the final word (e.g. trailing "[sad]")
            if session_start is not None:
                end_t = session_start + session_frames / self.backend.sample_rate
                trailing = [EmotionEvent(end_t, emo) for idx, emo in tags if idx >= word_index]
                if trailing:
                    self.schedule.append((), trailing)
                for idx, act in actions:
                    if idx >= word_index:
                        action_handles.append(self._fire_at(end_t, act))
            self.stats = {
                "time_to_first_audio_ms": None if first_audio_at is None else round((first_audio_at - t_request) * 1000),
                "words": word_index,
                "audio_seconds": round(session_frames / self.backend.sample_rate, 2),
            }
            if first_audio_at is not None:
                print(f"[speech] first audio {self.stats['time_to_first_audio_ms']} ms after request, "
                      f"{word_index} words, {self.stats['audio_seconds']} s audio")
        except asyncio.CancelledError:
            for h in action_handles:           # interrupted: actions for unspoken words never fire
                h.cancel()
            raise
        finally:
            if not stripper.done():
                stripper.cancel()

    def _fire_at(self, when: float, act: Action) -> asyncio.TimerHandle:
        delay = max(0.0, when - self.audio.timeline_time())
        return self._loop.call_later(delay, self._fire, act)

    def _fire(self, act: Action) -> None:
        try:
            if self.on_action is None:
                print(f"[action] {act.raw}")
            else:
                self.on_action(act)
        except Exception as e:                 # an action must never break speech
            print(f"[action] {act.raw}: {e}")
