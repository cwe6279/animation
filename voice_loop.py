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
import sys
import threading
import time
from typing import Callable, Iterable, Iterator, Optional

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

    def __init__(self, stt: STTBackend, llm_reply: Callable[[str], Iterator[str]],
                 speaker: Speaker, barge_in: bool = False,
                 on_event: Optional[Callable[[str, str], None]] = None):
        self.stt = stt
        self.llm_reply = llm_reply
        self.speaker = speaker
        self.barge_in = barge_in
        self.on_event = on_event or (lambda kind, text: print(f"[{kind}] {text}"))
        self._lock = threading.Lock()
        self._thinking = False
        self._last_busy = 0.0
        self._partial = ""
        self.turns = 0
        self._last_audio_in = 0.0
        self._speech_end_at = 0.0     # when the STT said the utterance ended

    # ── mic path ────────────────────────────────────────
    def process(self, pcm: bytes) -> None:
        self._last_audio_in = time.monotonic()
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
        if t is None:
            return
        if not t.final:
            if t.text != self._partial:
                self._partial = t.text
                self.on_event("hearing", t.text)
            return
        self._partial = ""
        if t.text.strip():
            self._speech_end_at = time.monotonic() - getattr(self.stt, "endpoint_delay_s", 0.0)
            self.on_user_text(t.text.strip())

    # ── one turn ────────────────────────────────────────
    def on_user_text(self, text: str) -> None:
        """Handle a finished user utterance (also used by --text-only)."""
        with self._lock:
            if self._thinking:
                return
            self._thinking = True
        self.turns += 1
        self.on_event("you", text)
        threading.Thread(target=self._answer, args=(text,), daemon=True, name="llm-turn").start()

    def _answer(self, text: str) -> None:
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
                    self.on_event("turn", f"you stopped -> transcript {round((t_end - t_stop) * 1000)} ms"
                                  f" -> first token {tok if tok is not None else '?'} ms"
                                  f" -> first audio {round((fa - t_stop) * 1000)} ms")
                    return
                time.sleep(0.02)
        threading.Thread(target=report_when_audio_starts, daemon=True).start()

        def timed_chunks():
            reply = []
            try:
                for chunk in self.llm_reply(text):
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
                self.on_event("bot", "".join(reply).strip())


        self.speaker.speak_stream(timed_chunks())


# ═══════════════════════════════════════════════════════
# MIC TOOLS  (--list-devices / --mic-test)
# ═══════════════════════════════════════════════════════
def stt_kwargs(args) -> dict:
    if args.stt == "vosk":
        return {"model_path": args.vosk_model, "silence_ms": args.silence_ms}
    if args.stt == "whisper":
        return {"model_size": args.whisper_model, "silence_ms": args.silence_ms}
    if args.stt in ("elevenlabs", "groq", "openai"):
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
    p.add_argument("--face", default="eve")
    p.add_argument("--face-dir", default=None)
    p.add_argument("--stt", default="whisper",
                   help="whisper (default, local, accurate) | vosk (local, light) | "
                        "elevenlabs (cloud Scribe realtime, best for a Pi) | groq | openai (cloud batch)")
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
    p.add_argument("--tts-model", default=None,
                   help="ElevenLabs model: eleven_v3 (default; performs [sigh]/[excited]-style tags) "
                        "or eleven_flash_v2_5 (~0.5 s faster, tags stripped)")
    p.add_argument("--llm", default="claude", choices=["claude", "groq", "openai"],
                   help="Which brain answers: claude (default), groq (Llama on Groq), openai")
    p.add_argument("--model", default=None,
                   help="Model id for the chosen --llm (defaults: claude-opus-5, qwen/qwen3.8-27b on Groq, gpt-4o-mini)")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--no-thinking", action="store_true",
                   help="Skip Claude's reasoning pass: faster first token, slightly less considered replies")
    p.add_argument("--character", default=None, help='Persona, e.g. "EVE from WALL-E, terse and curious"')
    p.add_argument("--barge-in", action="store_true", help="Interrupt playback when you start talking")
    p.add_argument("--text-only", action="store_true", help="Type in the window instead of using the mic")
    p.add_argument("--debug", "-d", action="store_true")
    p.add_argument("--no-hud", action="store_true", help="Hide key hints and text box")
    p.add_argument("--fullscreen", action="store_true",
                   help="Projection mode: fullscreen, face scaled to the display, no overlay or cursor (F toggles)")
    p.add_argument("--sync-offset", type=float, default=0.0)
    p.add_argument("--no-audio", action="store_true", help="No sound device (implies --text-only)")
    args = p.parse_args(argv)

    import pygame
    from face_asset_loader import FaceAssetLoader, default_manifest
    from talker import TalkerApp, build_audio, resolve_face_dir
    from tts_backends import make_backend
    from llm_integration.claude_chat import ClaudeChat

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
    try:
        backend = make_backend(args.tts, voice=voice, model=tts_model)
    except Exception as e:
        print(f"[error] TTS backend '{args.tts}' unavailable: {e}")
        return 1
    audio = build_audio(args.no_audio, args.sync_offset, args.output_device)
    text_only = args.text_only or args.no_audio

    if args.llm == "claude":
        chat = ClaudeChat(model=args.model or "claude-opus-5", effort=args.effort, character=character,
                          thinking=not args.no_thinking)
    else:
        from llm_integration.openai_compat_chat import OpenAICompatChat
        chat = (OpenAICompatChat.groq if args.llm == "groq" else OpenAICompatChat.openai)(
            model=args.model, character=character)
    print(f"[voice] brain: {args.llm} {chat.model}")
    app = TalkerApp(assets, audio, backend, debug=args.debug, show_hud=not args.no_hud,
                    fullscreen=args.fullscreen)

    stt = None
    if not text_only:
        try:
            from stt_backends import make_stt
            stt = make_stt(args.stt, **stt_kwargs(args))
        except Exception as e:
            print(f"[error] STT backend '{args.stt}' unavailable: {e}")
            return 1

    loop = VoiceLoop(stt, chat.reply, app, barge_in=args.barge_in)
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

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
