"""
talker.py — Phoneme-synced animated face
========================================
Input:   --text "..."  |  --file audio.wav  |  --mic  |  type in the window
Output:  pygame window (live display)

Speech goes through a streaming pipeline (speech_pipeline.SpeechPipeline):
sentences are synthesized one at a time and start playing while the rest
of the text is still being generated, so an LLM can stream its answer
straight into speak_stream().

Keys:  Enter/T = focus text box   Esc = unfocus / quit   D = debug overlay
       Ctrl+C in the box = stop speaking
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Callable, Iterable, Optional

import pygame

from audio_engine import BaseAudioEngine, NullAudioEngine
from face_asset_loader import AssetFaceRenderer, FaceAssetLoader, LoadedFaceAssets, default_manifest
from phoneme_scheduler import Emotion, ScheduleReader, Viseme, parse_emotion
from speech_pipeline import SpeechPipeline
from tts_backends import TTSBackend, make_backend


# ═══════════════════════════════════════════════════════
# AMPLITUDE FALLBACK (mic mode / raw audio files)
# ═══════════════════════════════════════════════════════
def amp_to_viseme(rms: float) -> Viseme:
    if rms < 120:
        return Viseme.SIL
    if rms < 600:
        return Viseme.DD
    if rms < 1800:
        return Viseme.AH
    return Viseme.AA


# ═══════════════════════════════════════════════════════
# IN-WINDOW TEXT BOX
# ═══════════════════════════════════════════════════════
class TextBox:
    """Single-line input along the bottom edge. Enter submits, Esc unfocuses."""

    def __init__(self, font: pygame.font.Font, width: int, height: int):
        self.font = font
        self.rect = pygame.Rect(16, height - 52, width - 32, 36)
        self.text = ""
        self.focused = False
        self.history: list[str] = []
        self._hist_idx = -1
        self._caret_blink = 0.0
        self._last_render: tuple = ()
        self._surf: Optional[pygame.Surface] = None

    def focus(self, on: bool = True) -> None:
        if on and not self.focused:
            pygame.key.start_text_input()
            pygame.key.set_text_input_rect(self.rect)
        elif not on and self.focused:
            pygame.key.stop_text_input()
        self.focused = on

    def handle(self, event: pygame.event.Event) -> Optional[str]:
        """Returns submitted text on Enter, else None."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.focus(self.rect.collidepoint(event.pos))
            return None
        if not self.focused:
            return None
        if event.type == pygame.TEXTINPUT:
            self.text += event.text
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                submitted = self.text.strip()
                self.text = ""
                self._hist_idx = -1
                if submitted:
                    self.history.append(submitted)
                    return submitted
            elif event.key == pygame.K_BACKSPACE:
                if event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT):
                    self.text = self.text.rstrip().rsplit(" ", 1)[0] if " " in self.text.rstrip() else ""
                else:
                    self.text = self.text[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.focus(False)
            elif event.key == pygame.K_UP and self.history:
                self._hist_idx = min(self._hist_idx + 1, len(self.history) - 1)
                self.text = self.history[-1 - self._hist_idx]
            elif event.key == pygame.K_DOWN and self.history:
                self._hist_idx = max(self._hist_idx - 1, -1)
                self.text = "" if self._hist_idx < 0 else self.history[-1 - self._hist_idx]
            elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                try:
                    self.text += pygame.scrap.get_text() or ""
                except Exception:
                    pass
        return None

    def draw(self, screen: pygame.Surface, dt: float) -> None:
        self._caret_blink = (self._caret_blink + dt) % 1.0
        caret = self.focused and self._caret_blink < 0.5
        key = (self.text, self.focused, caret)
        if key != self._last_render or self._surf is None:
            self._last_render = key
            s = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            s.fill((20, 20, 20, 200 if self.focused else 120))
            pygame.draw.rect(s, (90, 90, 90) if self.focused else (50, 50, 50), s.get_rect(), 1)
            if self.text or self.focused:
                label = self.font.render(self.text + ("|" if caret else ""), True, (230, 230, 230))
                # keep the tail visible when the line overflows
                x = min(8, self.rect.width - 8 - label.get_width())
                s.blit(label, (x, (self.rect.height - label.get_height()) // 2))
            else:
                hint = self.font.render("Press Enter, type text, Enter again to speak", True, (90, 90, 90))
                s.blit(hint, (8, (self.rect.height - hint.get_height()) // 2))
            self._surf = s
        screen.blit(self._surf, self.rect.topleft)


# ═══════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════
class TalkerApp:
    def __init__(self, assets: LoadedFaceAssets, audio: BaseAudioEngine,
                 backend: Optional[TTSBackend] = None, debug: bool = False,
                 default_emotion: Optional[str] = None, auto_exit: bool = False,
                 lead_seconds: float = 0.04, show_hud: bool = True, fullscreen: bool = False):
        self.debug = debug
        self.show_hud = show_hud and not fullscreen
        self.fullscreen = fullscreen
        self._auto_exit = auto_exit
        self._auto_exit_at: Optional[float] = None
        self._default_emotion = parse_emotion(default_emotion)
        if default_emotion and self._default_emotion is None:
            print(f"[warn] Unknown emotion '{default_emotion}', using neutral")

        m = assets.manifest
        self._w, self._h, self._fps = m.canvas_w, m.canvas_h, max(1, m.fps)
        self.renderer = AssetFaceRenderer(assets)

        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        self.screen = self._open_display()
        pygame.display.set_caption(f"Talker — {m.name}")
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 18)
        self._hud = [self.font_sm.render(t, True, (55, 55, 55))
                     for t in ("Enter=type & speak", "D=debug  F=fullscreen  H=hide", "ESC=quit")]
        self.textbox = TextBox(self.font_sm, self._w, self._h)

        self.audio = audio
        self.schedule = ScheduleReader()
        self.pipeline: Optional[SpeechPipeline] = None
        if backend is not None:
            self.pipeline = SpeechPipeline(audio, self.schedule, backend, lead_seconds=lead_seconds)
            self.pipeline.start()
        self._mic_mode = False
        self._amplitude_mode = False
        self.running = True
        # Optional hook: typed text goes here instead of straight to speak()
        # (voice_loop.py routes it through the LLM).
        self.on_submit: Optional[Callable[[str], None]] = None

    def _open_display(self) -> pygame.Surface:
        """
        Windowed: an exact canvas-sized window. Fullscreen (projection): SDL
        scales the canvas to the display, keeping aspect with black bars, so
        the face fills the projector without changing any face coordinates.
        """
        if self.fullscreen:
            flags = pygame.FULLSCREEN | pygame.SCALED
            pygame.mouse.set_visible(False)
        else:
            flags = 0
            pygame.mouse.set_visible(True)
        return pygame.display.set_mode((self._w, self._h), flags)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.show_hud = not self.fullscreen
        self.screen = self._open_display()

    # ── speech entry points ───────────────────────────────
    def speak(self, text: str) -> None:
        if self.pipeline is None:
            print("[talker] no TTS backend configured")
            return
        self._amplitude_mode = False
        print(f"[talker] speak: {text!r}")
        self.pipeline.speak(text)

    def speak_stream(self, chunks: Iterable[str]) -> None:
        """Feed text incrementally, e.g. tokens from an LLM."""
        if self.pipeline is None:
            print("[talker] no TTS backend configured")
            return
        self._amplitude_mode = False
        self.pipeline.speak_stream(chunks)

    def interrupt(self) -> None:
        if self.pipeline is not None:
            self.pipeline.interrupt()

    @property
    def is_busy(self) -> bool:
        """True while speech is being synthesized or its audio is still playing."""
        return self.pipeline is not None and self.pipeline.is_busy

    def play_file(self, path: str) -> None:
        """Play a WAV with amplitude-driven mouth (no phoneme schedule)."""
        self._amplitude_mode = True
        stop = threading.Event()

        def run():
            try:
                self.audio.open()
                _, end = self.audio.play_wav(path, stop_event=stop)
                if self._auto_exit:
                    self._auto_exit_at = time.monotonic() + max(0.0, end - self.audio.timeline_time()) + 0.8
            except Exception as e:
                print(f"[audio] cannot play {path}: {e}")
                if self._auto_exit:
                    self._auto_exit_at = time.monotonic()
        threading.Thread(target=run, daemon=True, name="play-file").start()

    def mic_mode(self) -> None:
        self._mic_mode = True
        self._amplitude_mode = True
        self.audio.start_mic()

    # ── main loop ────────────────────────────────────────
    def run(self) -> None:
        prev = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                dt = min(now - prev, 0.05)
                prev = now
                self._handle_events()

                if self._auto_exit and self._auto_exit_at is None and self.pipeline is not None \
                        and not self._amplitude_mode and self.pipeline.last_error:
                    self._auto_exit_at = now          # TTS failed: don't hang the demo
                if self._auto_exit and self._auto_exit_at is None and self.pipeline is not None \
                        and not self._amplitude_mode and self.pipeline.speech_end_time > 0 \
                        and not self.pipeline.is_busy:
                    self._auto_exit_at = now + 0.8    # let the mouth settle
                if self._auto_exit_at is not None and now >= self._auto_exit_at:
                    break

                rms, t = self.audio.get_state()
                if self._amplitude_mode or self._mic_mode:
                    viseme, emotion = amp_to_viseme(rms), Emotion.NEUTRAL
                else:
                    viseme = self.schedule.current_viseme(t)
                    emotion = self.schedule.current_emotion(t)
                if self._default_emotion and emotion is Emotion.NEUTRAL:
                    emotion = self._default_emotion

                self.renderer.update(viseme, dt, emotion)
                self.renderer.draw(self.screen)
                if self.debug:
                    self._draw_debug(rms, t, viseme)
                if self.show_hud:
                    for i, s in enumerate(self._hud):
                        self.screen.blit(s, (10, 10 + i * 22))
                    self.textbox.draw(self.screen, dt)
                pygame.display.flip()
                self.clock.tick(self._fps)
                if t > 600:      # keep memory flat in long sessions
                    self.schedule.trim_before(t - 60)
        finally:
            self.close()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL:
                self.interrupt()
                continue
            submitted = self.textbox.handle(event)
            if submitted:
                (self.on_submit or self.speak)(submitted)
                continue
            if self.textbox.focused:
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_t):
                    self.textbox.focus(True)
                elif event.key == pygame.K_d:
                    self.debug = not self.debug
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif event.key in (pygame.K_f, pygame.K_F11):
                    self.toggle_fullscreen()

    def _draw_debug(self, rms: float, t: float, viseme: Viseme) -> None:
        r = self.renderer
        p = self.pipeline
        lines = [
            f"RMS:     {rms:6.0f}",
            f"Time:    {t:8.3f}s",
            f"Viseme:  {viseme.value}",
            f"Shape:   {r.current_viseme_label}",
            f"Emotion: {r.current_emotion_label}",
            f"Open:    {r._open:.2f}",
            f"Width:   {r._width_t:.2f}",
            f"Round:   {r._rounded_blend:.2f}",
            f"FPS:     {self.clock.get_fps():.0f}",
            f"Queued:  {self.audio.queued_seconds():.2f}s",
        ]
        if p is not None and p.stats.get("time_to_first_audio_ms") is not None:
            lines.append(f"TTFA:    {p.stats['time_to_first_audio_ms']} ms")
        x = self._w - 250
        for i, line in enumerate(lines):
            self.screen.blit(self.font_sm.render(line, True, (0, 220, 80)), (x, 10 + i * 22))

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.audio.close()
        pygame.quit()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════
def resolve_face_dir(face: str, face_dir: Optional[str]) -> Optional[str]:
    if face_dir:
        return face_dir
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "faces", face)
    return candidate if os.path.isfile(os.path.join(candidate, "face.json")) else None


def build_audio(no_audio: bool, sync_offset: float, output_device: Optional[int] = None) -> BaseAudioEngine:
    if no_audio:
        return NullAudioEngine(sync_offset=sync_offset)
    try:
        from audio_engine import AudioEngine
        return AudioEngine(sync_offset=sync_offset, output_device=output_device)
    except Exception as e:
        print(f"[audio] no output device ({e}); running silent")
        return NullAudioEngine(sync_offset=sync_offset)


def main(argv=None) -> int:
    from env_config import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Talker — phoneme-synced animated face")
    parser.add_argument("--text", "-t", type=str, help="Speak this text (supports [emotion] tags)")
    parser.add_argument("--file", "-f", type=str, help="Play a WAV file (amplitude-driven mouth)")
    parser.add_argument("--mic", "-m", action="store_true", help="Live microphone (amplitude mode)")
    parser.add_argument("--stdin", action="store_true",
                        help="Read text from stdin as it arrives (pipe an LLM into it)")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--no-hud", action="store_true", help="Hide key hints and text box")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Projection mode: fullscreen, face scaled to the display, no overlay or cursor (F toggles)")
    parser.add_argument("--face", type=str, default="pumpkin", help="Face name (folder under faces/)")
    parser.add_argument("--face-dir", type=str, default=None, help="Path to a face directory (overrides --face)")
    parser.add_argument("--emotion", type=str, default=None,
                        help="Default emotion: neutral, happy, angry, annoyed, sad, surprise")
    parser.add_argument("--auto-exit", action="store_true", help="Exit after speech finishes (for scripts)")
    parser.add_argument("--tts", type=str, default="edge", help="TTS backend: edge (default) or elevenlabs")
    parser.add_argument("--voice", type=str, default=None,
                        help="Voice name/id for the backend (edge: en-US-GuyNeural, elevenlabs: voice id)")
    parser.add_argument("--tts-model", type=str, default=None,
                        help="ElevenLabs model: eleven_flash_v2_5 (default) or eleven_v3 (performs audio tags)")
    parser.add_argument("--sync-offset", type=float, default=0.0,
                        help="Seconds to shift the face relative to audio (+ later, - earlier)")
    parser.add_argument("--lead", type=float, default=0.04,
                        help="Seconds mouth shapes lead their sound to offset easing lag")
    parser.add_argument("--no-audio", action="store_true", help="Render without a sound device")
    parser.add_argument("--output-device", default=None,
                        help='Output device: name fragment ("jabra") or index (see voice_loop.py --list-devices)')
    args = parser.parse_args(argv)

    # Display + font only: SDL's own audio subsystem is not used (PortAudio is)
    # and initialising it can fight with the device or hang on exit.
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)   # so asset loading can convert_alpha()

    face_dir = resolve_face_dir(args.face, args.face_dir)
    try:
        if face_dir:
            assets = FaceAssetLoader().load(face_dir)
        else:
            print(f"[assets] no faces/{args.face}/face.json — using procedural default")
            assets = FaceAssetLoader().build(default_manifest(args.face))
    except Exception as e:
        print(f"[error] Could not load face: {e}")
        return 1

    backend = None
    if not (args.file or args.mic):
        m = assets.manifest
        try:
            backend = make_backend(args.tts, voice=args.voice or m.voices.get(args.tts.lower()),
                                   model=args.tts_model or m.tts_model)
        except Exception as e:
            print(f"[error] TTS backend '{args.tts}' unavailable: {e}")
            return 1

    audio = build_audio(args.no_audio, args.sync_offset, args.output_device)
    app = TalkerApp(assets, audio, backend, debug=args.debug, default_emotion=args.emotion,
                    auto_exit=args.auto_exit, lead_seconds=args.lead, show_hud=not args.no_hud,
                    fullscreen=args.fullscreen)

    if args.text:
        app.speak(args.text)
    elif args.stdin:
        def lines():
            for line in sys.stdin:
                yield line
        app.speak_stream(lines())
    elif args.file:
        app.play_file(args.file)
    elif args.mic:
        try:
            app.mic_mode()
        except Exception as e:
            print(f"[error] mic unavailable: {e}")
            return 1
    else:
        print("No input — press Enter in the window and type, or use --text / --stdin / --file / --mic")

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
