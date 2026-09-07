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
import queue
import re
import sys
import threading
import time
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

from stt_backends import STTBackend, Transcript


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

    def __init__(self, stt: STTBackend, llm_reply: Callable[[str], Iterator[str]],
                 speaker: Speaker, barge_in: bool = False,
                 on_event: Optional[Callable[[str, str], None]] = None,
                 wake_words: Optional[List[str]] = None, idle_timeout: float = 45.0,
                 clock=time.monotonic, start_engaged: bool = True):
        self.stt = stt
        self.llm_reply = llm_reply
        self.speaker = speaker
        self.barge_in = barge_in
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
        self._last_activity = self.clock()
        self._lock = threading.Lock()
        self._thinking = False
        self._last_busy = 0.0
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
        self._was_speaking = False
        self._vision_ticket: Optional[int] = None

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
        self.tick()
        busy = self.speaker.is_busy or self._thinking
        now = time.monotonic()
        if busy:
            self._last_busy = now
            if not self.barge_in or self._thinking:
                self.stt.reset()          # drop echo; nothing to transcribe
                return
            t = self.stt.feed(pcm)
            if self.stt.speech_active or (t and t.text):
                self.on_event("barge-in", t.text if t else "")
                self.speaker.interrupt()
                self._last_busy = 0.0
            return
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
                self.on_event("hearing", t.text)
            return
        self._partial = ""
        if t.text.strip():
            self._speech_end_at = (time.monotonic() - getattr(self.stt, "endpoint_delay_s", 0.0)
                                   - getattr(self.stt, "last_transcribe_s", 0.0))
            self.on_user_text(t.text.strip())

    # ── wake mode ───────────────────────────────────────
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

    def tick(self) -> None:
        """Idle timeout check; called per audio chunk."""
        if self.engaged and self.wake_words and not self.speaker.is_busy and not self._thinking:
            if self.clock() - self._last_activity > self.idle_timeout:
                self.disengage(f"quiet for {self.idle_timeout:.0f}s")

    # ── one turn ────────────────────────────────────────
    def on_user_text(self, text: str) -> None:
        """Handle a finished user utterance (also used by --text-only)."""
        if self.wake_words:
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
        with self._lock:
            if self._thinking:
                return
            self._thinking = True
        self.turns += 1
        self.on_event("you", text)
        threading.Thread(target=self._answer, args=(text,), daemon=True, name="llm-turn").start()

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
        # A visual question: give the in-flight burst time to land first (capture ~0.5 s
        # + vision model ~2.3 s, minus what already elapsed while the visitor spoke).
        if self.vision is not None and self._vision_ticket is not None and self.VISUAL_RE.search(text):
            self.vision.wait_for(self._vision_ticket, timeout=2.5)
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
    import os
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
    from audio_engine import AudioEngine
    audio = AudioEngine()
    print("Input devices:")
    for idx, name, rate, is_default in audio.list_input_devices():
        print(f"  [{idx}] {name}  ({rate} Hz){'  <- default' if is_default else ''}")
    if args.list_devices:
        print("Output devices:")
        for idx, name, rate, is_default in audio.list_output_devices():
            print(f"  [{idx}] {name}  ({rate} Hz){'  <- default' if is_default else ''}")
        audio.close()
        return 0

    from stt_backends import make_stt, EnergyEndpointer
    stt = make_stt(args.stt, **stt_kwargs(args))
    state = {"peak": 0.0, "last": "", "n": 0}
    recorder = None
    if args.record:
        import os as _os
        _os.makedirs(args.record, exist_ok=True)
        recorder = EnergyEndpointer(stt.sample_rate, silence_ms=args.silence_ms or 600)
        print(f"Recording utterances to {args.record}/ (WAV + transcripts.txt draft references)")

    def save_clip(audio: bytes, text: str):
        import wave
        state["n"] += 1
        name = f"utt_{state['n']:03d}.wav"
        with wave.open(_os.path.join(args.record, name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(stt.sample_rate); w.writeframes(audio)
        with open(_os.path.join(args.record, "transcripts.txt"), "a", encoding="utf-8") as f:
            f.write(f"{name}\t{text}\n")
        print(f"[saved]   {name} ({len(audio)/2/stt.sample_rate:.1f}s)")

    pending_clip = {"audio": None}

    def on_frames(pcm):
        import numpy as np
        s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
        state["peak"] = max(state["peak"], rms)
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
    print("Speak. Level is printed every 2 s (aim for 500-5000; below ~200 is too quiet). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(2)
            rms, _ = audio.get_state()
            print(f"\r[level]   now {rms:5.0f}  peak {state['peak']:5.0f}{' ':50}")
    except KeyboardInterrupt:
        pass
    finally:
        audio.close()
    return 0


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════
def main(argv=None) -> int:
    from env_config import load_dotenv
    load_dotenv()
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
    p.add_argument("--mic-test", action="store_true",
                   help="Only print what the mic hears (levels + transcripts); no Claude, no voice")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="With --mic-test: save each utterance as WAV in DIR plus transcripts.txt "
                        "(draft references to correct, then run bench_stt.py DIR)")
    p.add_argument("--tts", default=None,
                   help="elevenlabs (default when ELEVENLABS_API_KEY is set) or edge (free)")
    p.add_argument("--voice", default=None, help="TTS voice name/id")
    p.add_argument("--voice-speed", type=float, default=None,
                   help="Speaking rate multiplier, e.g. 1.15 (ElevenLabs Flash and edge honour it; v3 ignores it)")
    p.add_argument("--tts-model", default=None,
                   help="ElevenLabs model: v3 (default; performs [sigh]/[excited]-style tags, ~1 s to first audio) "
                        "or flash (~0.25 s, tags stripped). Full model ids also accepted.")
    p.add_argument("--llm", default="claude", choices=["claude", "openai"],
                   help="Which brain answers: claude (default) or openai (for comparison)")
    p.add_argument("--model", default=None,
                   help="Model id for the chosen --llm (defaults: claude-opus-5, gpt-4o-mini)")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--thinking", action="store_true",
                   help="Enable Claude's reasoning pass before answering (about +1 s to first token; off by default)")
    p.add_argument("--no-thinking", action="store_true", help=argparse.SUPPRESS)   # kept for old scripts
    p.add_argument("--character", default=None, help='Persona, e.g. "EVE from WALL-E, terse and curious"')
    p.add_argument("--barge-in", action="store_true", help="Interrupt playback when you start talking")
    p.add_argument("--text-only", action="store_true", help="Type in the window instead of using the mic")
    p.add_argument("--debug", "-d", action="store_true")
    p.add_argument("--no-hud", action="store_true", help="Hide key hints and text box")
    p.add_argument("--fullscreen", action="store_true",
                   help="Projection mode: fullscreen, face scaled to the display, no overlay or cursor (F toggles)")
    p.add_argument("--sync-offset", type=float, default=0.0)
    p.add_argument("--no-audio", action="store_true", help="No sound device (implies --text-only)")
    p.add_argument("--wake", action="store_true",
                   help="Wake mode: stay dormant until the character's name (or --wake-word) is heard; "
                        "go dormant again after --idle-timeout seconds of silence or when the chat ends")
    p.add_argument("--wake-word", default=None,
                   help='Comma-separated wake words (implies --wake), e.g. "eve, hey eve". '
                        "Default: the face's wake_words, else its name")
    p.add_argument("--idle-timeout", type=float, default=45.0,
                   help="Seconds of silence before returning to dormant in wake mode (default 45)")
    p.add_argument("--start-dormant", action="store_true",
                   help="In wake mode, start dormant instead of engaged (kiosk: wait to be called by name)")
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
    args = p.parse_args(argv)
    if args.profile == "pi":
        apply_pi_profile(args)

    import pygame
    from face_asset_loader import FaceAssetLoader, default_manifest
    from talker import TalkerApp, build_audio, resolve_face_dir
    from tts_backends import make_backend
    from llm_integration.claude_chat import ClaudeChat

    if args.list_cameras:
        from vision import list_cameras
        cams = list_cameras()
        print("Cameras:" if cams else "No cameras found")
        for c in cams:
            print(f"  [{c['index']}] {c['name']}  ({c['path']})")
        return 0
    if args.list_devices or args.mic_test:
        return mic_tools(args)

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)
    face_dir = resolve_face_dir(args.face, args.face_dir)
    assets = FaceAssetLoader().load(face_dir) if face_dir else FaceAssetLoader().build(default_manifest(args.face))

    # Face-level defaults for voice, model and persona (flags win)
    m = assets.manifest
    import os
    args.tts = (args.tts or ("elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "edge")).lower()
    voice = args.voice or m.voices.get(args.tts)
    tts_model = args.tts_model or m.tts_model or ("eleven_v3" if args.tts == "elevenlabs" else None)
    character = args.character or m.character or None
    speed = args.voice_speed or m.voice_speed
    try:
        backend = make_backend(args.tts, voice=voice, model=tts_model, speed=speed)
    except Exception as e:
        print(f"[error] TTS backend '{args.tts}' unavailable: {e}")
        return 1
    audio = build_audio(args.no_audio, args.sync_offset, args.output_device)
    text_only = args.text_only or args.no_audio

    can_see = args.camera is not None and not args.no_vision
    wake_words = None
    if args.wake or args.wake_word:
        wake_words = ([w for w in args.wake_word.split(",")] if args.wake_word
                      else (m.wake_words or [m.name.replace("_", " ")]))
    if args.llm == "claude":
        chat = ClaudeChat(model=args.model or "claude-opus-5", effort=args.effort, character=character,
                          thinking=bool(args.thinking), can_see=can_see, wake_mode=bool(wake_words))
    else:
        from llm_integration.openai_compat_chat import OpenAICompatChat
        chat = OpenAICompatChat.openai(model=args.model, character=character, can_see=can_see,
                                       wake_mode=bool(wake_words))
    print(f"[voice] brain: {args.llm} {chat.model}{' (told it can see)' if can_see else ''}")
    app = TalkerApp(assets, audio, backend, debug=args.debug, show_hud=not args.no_hud,
                    fullscreen=args.fullscreen, adaptive_fps=not args.fixed_fps)

    stt = None
    if not text_only:
        try:
            from stt_backends import make_stt
            stt = make_stt(args.stt, **stt_kwargs(args))
        except Exception as e:
            print(f"[error] STT backend '{args.stt}' unavailable: {e}")
            return 1

    watcher = None
    if args.camera is not None and not args.no_vision:
        try:
            from vision import CameraSource, SceneWatcher, describe_with_claude, resolve_camera
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

    loop = VoiceLoop(stt, chat.reply, app, barge_in=args.barge_in, wake_words=wake_words,
                     idle_timeout=args.idle_timeout, start_engaged=not args.start_dormant)
    loop.vision = watcher
    if wake_words:
        names = ", ".join(repr(w) for w in loop.wake_words)
        print(f"[mode] {'dormant, listening for ' + names if not loop.engaged else 'engaged; after ' + str(int(args.idle_timeout)) + 's of quiet, wakes on ' + names}")
    app.on_submit = loop.on_user_text        # typed text goes through Claude too

    if stt is not None:
        try:
            audio.start_mic(on_frames=loop.process, rate=stt.sample_rate,
                            device=args.mic_device, open_rate=args.mic_rate)
        except Exception as e:
            print(f"[error] mic unavailable: {e}")
            return 1
        print(f"[voice] listening ({stt.name}) — talk to the face. Esc quits."
              + (f"  voice={voice}" if voice else "") + (f"  character={character!r}" if character else ""))
    else:
        print("[voice] text-only: press Enter in the window, type, Enter to send to Claude.")

    try:
        app.run()
    finally:
        if watcher is not None:
            watcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
