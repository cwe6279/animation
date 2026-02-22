"""
talker.py  v2 — Phoneme-synced animated face
============================================
Input:   --text "..."  |  --file audio.wav  |  --mic
Output:  pygame window (live display)

Face is fully swappable via FaceConfig.
Lip sync uses timed viseme schedule built from edge-tts word
timestamps + g2p phoneme expansion. Falls back to amplitude
mode for mic input or when TTS libs are unavailable.
"""

import pygame
import pyaudio
import numpy as np
import math
import threading
import time
import wave
import sys
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

from phoneme_scheduler import (
    Viseme, VisemeProps, VISEME_PROPS,
    VisemeEvent, PhonemeScheduler, ScheduleReader,
    Emotion
)
from face_asset_loader import (
    FaceAssetLoader, AssetFaceRenderer, LoadedFaceAssets
)


# ═══════════════════════════════════════════════════════
# FACE CONFIG
# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
# FACE CONFIG
# ═══════════════════════════════════════════════════════
@dataclass
class FaceConfig:
    name: str = "pumpkin"
    width:  int = 800
    height: int = 800
    fps:    int = 60

    bg_color:    Tuple = (0, 0, 0)
    shape_color: Tuple = (255, 200, 0)    # bright yellow fill
    glow_color:  Tuple = (255, 130, 0)    # orange outer glow
    glow_dark:   Tuple = (180, 60, 0)     # amber inner corner shadow

    left_eye_cx:  int = 240
    left_eye_cy:  int = 290
    right_eye_cx: int = 560
    right_eye_cy: int = 290
    eye_w: int = 110
    eye_h: int = 100

    mouth_cx: int = 400
    mouth_cy: int = 520
    mouth_max_w: int = 320
    mouth_min_w: int = 200
    n_teeth:    int = 4


# ═══════════════════════════════════════════════════════
# FACE RENDERER — minimal glowing cuts on black
# Triangle eyes + toothed mouth, yellow fill,
# orange halo glow, amber corner drop shadow
# ═══════════════════════════════════════════════════════
class FaceRenderer:
    def __init__(self, cfg: FaceConfig):
        self.cfg = cfg
        self._open           = 0.0
        self._width_t        = 0.8
        self._rounded_blend  = 0.0
        self._blink_t        = 0.0
        self._blink_next     = time.time() + np.random.uniform(2.0, 4.0)
        self.current_viseme_label = "sil"
        self._gs = pygame.Surface((cfg.width, cfg.height), pygame.SRCALPHA)

    def update(self, viseme: Viseme, dt: float):
        props = VISEME_PROPS[viseme]
        self.current_viseme_label = props.label

        spd_open  = 20.0 if props.open_amount > self._open   else 9.0
        spd_width = 16.0 if props.width_scale > self._width_t else 8.0

        self._open    += (props.open_amount - self._open)    * spd_open  * dt
        self._width_t += (props.width_scale - self._width_t) * spd_width * dt
        self._open    = max(0.0, min(1.0, self._open))
        self._width_t = max(0.0, min(1.0, self._width_t))

        target_round = 1.0 if props.rounded else 0.0
        self._rounded_blend += (target_round - self._rounded_blend) * 14.0 * dt

        now = time.time()
        if now >= self._blink_next:
            self._blink_t    = 1.0
            self._blink_next = now + np.random.uniform(2.5, 5.5)
        if self._blink_t > 0:
            self._blink_t = max(0.0, self._blink_t - dt * 9.0)

    def draw(self, surf: pygame.Surface):
        surf.fill(self.cfg.bg_color)
        self._draw_eye(surf, self.cfg.left_eye_cx,  self.cfg.left_eye_cy)
        self._draw_eye(surf, self.cfg.right_eye_cx, self.cfg.right_eye_cy)
        self._draw_mouth(surf)

    def _draw_eye(self, surf, cx, cy):
        cfg    = self.cfg
        ew, eh = cfg.eye_w, cfg.eye_h
        eh_now = max(3, int(eh * (1.0 - self._blink_t * 0.97)))
        tip = (cx,           cy - eh_now // 2)
        bl  = (cx - ew // 2, cy + eh_now // 2)
        br  = (cx + ew // 2, cy + eh_now // 2)
        pts = [tip, bl, br]
        self._draw_glow(surf, pts, cfg.glow_color, layers=6, spread=14)
        pygame.draw.polygon(surf, cfg.shape_color, pts)
        self._draw_corner_shadow(surf, br, cfg.glow_dark)

    def _draw_mouth(self, surf):
        cfg    = self.cfg
        cx, cy = cfg.mouth_cx, cfg.mouth_cy
        oa     = self._open
        rb     = self._rounded_blend
        w      = int(cfg.mouth_min_w + (cfg.mouth_max_w - cfg.mouth_min_w) * self._width_t)
        open_h = int(110 * oa)

        if oa < 0.04:
            pts = [(cx-w//2, cy-7), (cx+w//2, cy-7),
                   (cx+w//2, cy+7), (cx-w//2, cy+7)]
            self._draw_glow(surf, pts, cfg.glow_color, layers=4, spread=10)
            pygame.draw.polygon(surf, cfg.shape_color, pts)
            self._draw_corner_shadow(surf, (cx+w//2, cy+7), cfg.glow_dark)
            return

        if rb >= 0.5:
            self._draw_oval_mouth(surf, cx, cy, w, open_h)
        else:
            self._draw_toothed_mouth(surf, cx, cy, w, open_h, oa)

    def _draw_toothed_mouth(self, surf, cx, cy, w, open_h, oa):
        cfg     = self.cfg
        teeth_h = int(28 * oa)
        n       = cfg.n_teeth
        top_pts, bot_pts = [], []
        for i in range(n * 2 + 1):
            x = cx - w//2 + int(i * w / (n * 2))
            top_pts.append((x, cy - open_h//2 + (0 if i%2==0 else teeth_h)))
            bot_pts.append((x, cy + open_h//2 - (0 if i%2==0 else teeth_h)))
        all_pts = top_pts + list(reversed(bot_pts))
        self._draw_glow(surf, all_pts, cfg.glow_color, layers=5, spread=12)
        pygame.draw.polygon(surf, cfg.shape_color, all_pts)
        self._draw_corner_shadow(surf, bot_pts[-1], cfg.glow_dark)

    def _draw_oval_mouth(self, surf, cx, cy, w, open_h):
        cfg = self.cfg
        ow  = max(30, int(w * 0.55))
        oh  = max(20, open_h)
        pts = []
        for i in range(24):
            angle = 2 * math.pi * i / 24
            pts.append((int(cx + ow//2 * math.cos(angle)),
                        int(cy + oh//2 * math.sin(angle))))
        self._draw_glow(surf, pts, cfg.glow_color, layers=5, spread=12)
        pygame.draw.ellipse(surf, cfg.shape_color, (cx-ow//2, cy-oh//2, ow, oh))
        self._draw_corner_shadow(surf, (cx + ow//3, cy + oh//3), cfg.glow_dark)

    def _draw_glow(self, surf, pts, color, layers=6, spread=14):
        gs = self._gs
        gs.fill((0, 0, 0, 0))
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        for i in range(layers, 0, -1):
            scale = 1.0 + (i / layers) * (spread / 60.0)
            alpha = int(80 * (i / layers) ** 1.4)
            expanded = [(int(cx + (x-cx)*scale), int(cy + (y-cy)*scale)) for x,y in pts]
            pygame.draw.polygon(gs, (*color, alpha), expanded)
        surf.blit(gs, (0, 0))

    def _draw_corner_shadow(self, surf, corner, color):
        gs = self._gs
        gs.fill((0, 0, 0, 0))
        for r in range(44, 0, -8):
            alpha = int(120 * (1 - r / 44.0) ** 1.5)
            pygame.draw.circle(gs, (*color, alpha), corner, r)
        surf.blit(gs, (0, 0))


# ═══════════════════════════════════════════════════════
# CAT FACE RENDERER
# Green iris + black vertical slit pupil, small cat mouth
# floating on pure black. Same corner-shadow glow technique.
# ═══════════════════════════════════════════════════════
class CatFaceRenderer:
    BG          = (0,   0,   0)
    IRIS        = (40,  200, 30)
    IRIS_OUTER  = (20,  140, 10)
    IRIS_INNER  = (80,  255, 50)
    PUPIL       = (5,   8,   3)
    GLOW        = (30,  200, 20)
    GLOW_DARK   = (10,  80,  5)
    MOUTH_COLOR = (200, 60,  110)
    MOUTH_DARK  = (120, 20,  50)

    LEFT_EYE  = (250, 310)
    RIGHT_EYE = (550, 310)
    EYE_RX    = 72
    EYE_RY    = 55
    PUPIL_RX  = 11
    PUPIL_RY  = 46
    MOUTH_CX  = 400
    MOUTH_CY  = 530

    def __init__(self, w=800, h=800):
        self.w, self.h = w, h
        self._open          = 0.0
        self._width_t       = 0.8
        self._rounded_blend = 0.0
        self._blink_t       = 0.0
        self._blink_next    = time.time() + np.random.uniform(2.0, 4.0)
        self.current_viseme_label = "sil"
        self._gs = pygame.Surface((w, h), pygame.SRCALPHA)

    def update(self, viseme: Viseme, dt: float):
        props = VISEME_PROPS[viseme]
        self.current_viseme_label = props.label
        spd_open  = 20.0 if props.open_amount > self._open   else 9.0
        spd_width = 16.0 if props.width_scale > self._width_t else 8.0
        self._open    += (props.open_amount - self._open)    * spd_open  * dt
        self._width_t += (props.width_scale - self._width_t) * spd_width * dt
        self._open    = max(0.0, min(1.0, self._open))
        self._width_t = max(0.0, min(1.0, self._width_t))
        target_round = 1.0 if props.rounded else 0.0
        self._rounded_blend += (target_round - self._rounded_blend) * 14.0 * dt
        now = time.time()
        if now >= self._blink_next:
            self._blink_t    = 1.0
            self._blink_next = now + np.random.uniform(2.5, 5.5)
        if self._blink_t > 0:
            self._blink_t = max(0.0, self._blink_t - dt * 9.0)

    def draw(self, surf: pygame.Surface):
        surf.fill(self.BG)
        self._draw_eye(surf, *self.LEFT_EYE)
        self._draw_eye(surf, *self.RIGHT_EYE)
        self._draw_cat_mouth(surf)

    def _draw_eye(self, surf, cx, cy):
        rx = self.EYE_RX
        ry = self.EYE_RY
        squish = 1.0 - self._blink_t * 0.97
        ry_now = max(2, int(ry * squish))
        px_now = max(1, int(self.PUPIL_RX * squish))
        py_now = max(1, int(self.PUPIL_RY * squish))

        # Outer green glow halo
        gs = self._gs
        gs.fill((0, 0, 0, 0))
        for i in range(7, 0, -1):
            t   = i / 7
            erx = int(rx + 18 * t)
            ery = int(ry + 13 * t)
            pygame.draw.ellipse(gs, (*self.GLOW, int(70 * t**1.6)),
                (cx-erx, cy-ery, erx*2, ery*2))
        surf.blit(gs, (0, 0))

        # Iris layers
        pygame.draw.ellipse(surf, self.IRIS_OUTER,
            (cx-rx-2, cy-ry_now-2, (rx+2)*2, (ry_now+2)*2))
        pygame.draw.ellipse(surf, self.IRIS,
            (cx-rx, cy-ry_now, rx*2, ry_now*2))
        pygame.draw.ellipse(surf, self.IRIS_INNER,
            (cx-rx+8, cy-ry_now+5, (rx-8)*2, (ry_now-5)*2), 3)

        # Bottom-right arc shadow for depth
        gs.fill((0, 0, 0, 0))
        for ox, oy, alpha in [(6,5,60),(4,3,40),(2,2,25)]:
            pygame.draw.ellipse(gs, (*self.GLOW_DARK, alpha),
                (cx-rx+ox, cy-ry_now+oy, rx*2, ry_now*2))
        surf.blit(gs, (0, 0))

        # Vertical slit pupil
        pygame.draw.ellipse(surf, self.PUPIL,
            (cx-px_now, cy-py_now, px_now*2, py_now*2))

        # Catchlights
        pygame.draw.ellipse(surf, (255,255,255),
            (cx-rx//3-4, cy-ry_now//2+4, 12, 16))
        pygame.draw.circle(surf, (220,255,220),
            (cx+rx//4, cy-ry_now//3), 5)

    def _draw_cat_mouth(self, surf):
        cx, cy = self.MOUTH_CX, self.MOUTH_CY
        oa     = self._open
        rb     = self._rounded_blend
        if oa < 0.04:
            self._draw_w_closed(surf, cx, cy)
        elif rb >= 0.5:
            self._draw_oval(surf, cx, cy, oa)
        else:
            self._draw_w_open(surf, cx, cy, oa)

    def _draw_w_closed(self, surf, cx, cy):
        pts_l = [(cx-55, cy-4), (cx-32, cy+18), (cx, cy+4)]
        pts_r = [(cx, cy+4),    (cx+32, cy+18), (cx+55, cy-4)]
        gs = self._gs
        gs.fill((0,0,0,0))
        for pts in (pts_l, pts_r):
            pygame.draw.lines(gs, (*self.MOUTH_COLOR, 50), False, pts, 10)
        surf.blit(gs, (0,0))
        pygame.draw.lines(surf, self.MOUTH_COLOR, False, pts_l, 3)
        pygame.draw.lines(surf, self.MOUTH_COLOR, False, pts_r, 3)

    def _draw_w_open(self, surf, cx, cy, oa):
        drop  = int(50 * oa)
        width = int(50 + 40 * self._width_t)
        top_pts = [
            (cx-width-10, cy-4),
            (cx-width+14, cy+16),
            (cx,          cy+4),
            (cx+width-14, cy+16),
            (cx+width+10, cy-4),
        ]
        bot_pts = [
            (cx-width-5,  cy-4),
            (cx-width+5,  cy+drop+20),
            (cx,          cy+drop+30),
            (cx+width-5,  cy+drop+20),
            (cx+width+5,  cy-4),
        ]
        pygame.draw.polygon(surf, (8,3,5), top_pts + list(reversed(bot_pts)))
        gs = self._gs
        gs.fill((0,0,0,0))
        pygame.draw.lines(gs, (*self.MOUTH_COLOR, 60), False, top_pts, 8)
        pygame.draw.lines(gs, (*self.MOUTH_COLOR, 60), False, bot_pts, 8)
        surf.blit(gs, (0,0))
        pygame.draw.lines(surf, self.MOUTH_COLOR, False, top_pts, 3)
        pygame.draw.lines(surf, self.MOUTH_COLOR, False, bot_pts, 3)

    def _draw_oval(self, surf, cx, cy, oa):
        ow = int(40 + 20 * self._width_t)
        oh = int(25 + 30 * oa)
        gs = self._gs
        gs.fill((0,0,0,0))
        pygame.draw.ellipse(gs, (*self.MOUTH_COLOR, 55),
            (cx-ow//2-6, cy-oh//2-6, ow+12, oh+12))
        surf.blit(gs, (0,0))
        pygame.draw.ellipse(surf, (8,3,5), (cx-ow//2, cy-oh//2, ow, oh))
        pygame.draw.ellipse(surf, self.MOUTH_COLOR, (cx-ow//2, cy-oh//2, ow, oh), 3)




# ═══════════════════════════════════════════════════════
class AudioEngine:
    CHUNK = 512

    def __init__(self):
        self.pa         = pyaudio.PyAudio()
        self.rms        = 0.0
        self.playback_t = 0.0
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._done_cb   = None

    def get_state(self):
        with self._lock:
            return self.rms, self.playback_t

    def play_wav(self, wav_path: str, done_callback=None):
        self._stop.clear()
        self._done_cb = done_callback
        threading.Thread(target=self._play_loop, args=(wav_path,), daemon=True).start()

    def _play_loop(self, wav_path: str):
        try:
            wf = wave.open(wav_path, 'rb')
        except Exception as e:
            print(f"[audio] cannot open {wav_path}: {e}")
            return

        rate = wf.getframerate()
        nch  = wf.getnchannels()
        sw   = wf.getsampwidth()
        print(f"[audio] {wav_path}: rate={rate} ch={nch} width={sw}")

        stream = self.pa.open(
            format=self.pa.get_format_from_width(sw),
            channels=nch,
            rate=rate,
            output=True,
            frames_per_buffer=self.CHUNK
        )

        # Use wall clock from first write — reliable on all platforms
        start_t    = None
        bps        = sw * nch

        while not self._stop.is_set():
            data = wf.readframes(self.CHUNK)
            if not data:
                break
            if start_t is None:
                start_t = time.time()
                with self._lock:
                    self.playback_t = 0.0
            stream.write(data)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
            with self._lock:
                self.rms        = rms
                self.playback_t = time.time() - start_t

        stream.stop_stream()
        stream.close()
        wf.close()
        with self._lock:
            self.rms = 0.0
        if self._done_cb:
            self._done_cb()

    def start_mic(self):
        self._stop.clear()
        def cb(in_data, frame_count, time_info, status):
            samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
            with self._lock:
                self.rms = rms
            return (in_data, pyaudio.paContinue)
        self._mic_stream = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=22050,
            input=True, frames_per_buffer=self.CHUNK,
            stream_callback=cb
        )
        self._mic_stream.start_stream()

    def stop(self):
        self._stop.set()

    def close(self):
        self.stop()
        try: self.pa.terminate()
        except: pass


# ═══════════════════════════════════════════════════════
# AMPLITUDE FALLBACK (mic mode / no schedule)
# ═══════════════════════════════════════════════════════
def amp_to_viseme(rms: float) -> Viseme:
    if rms < 120:  return Viseme.SIL
    if rms < 600:  return Viseme.DD
    if rms < 1800: return Viseme.AH
    return Viseme.AA


# ═══════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════
class TalkerApp:
    def __init__(self, cfg: FaceConfig = None, assets: LoadedFaceAssets = None,
                 debug: bool = False, default_emotion: str = None,
                 auto_exit: bool = False):
        self.debug = debug
        self._auto_exit = auto_exit
        self._auto_exit_at = None   # time.time() when we should quit
        self._default_emotion = None
        if default_emotion:
            from phoneme_scheduler import _EMOTION_NAMES
            self._default_emotion = _EMOTION_NAMES.get(default_emotion.lower())
            if not self._default_emotion:
                print(f"[warn] Unknown emotion '{default_emotion}', using neutral")
        pygame.init()

        if assets:
            # Asset-driven face (from face directory)
            m = assets.manifest
            w, h = m.canvas_w, m.canvas_h
            name = m.name
            self.renderer = AssetFaceRenderer(assets)
        else:
            # Procedural face (from FaceConfig)
            cfg = cfg or FaceConfig()
            w, h = cfg.width, cfg.height
            name = cfg.name
            if cfg.name == "green_cat":
                self.renderer = CatFaceRenderer(w, h)
            else:
                self.renderer = FaceRenderer(cfg)

        os.environ['SDL_VIDEO_CENTERED'] = '1'
        self.screen   = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"Talker — {name}")
        self.clock    = pygame.time.Clock()
        self.audio    = AudioEngine()
        self.running  = True
        self.font_sm  = pygame.font.SysFont("monospace", 18)
        self._schedule_reader: Optional[ScheduleReader] = None
        self._mic_mode = False
        self._scheduler = PhonemeScheduler()
        self._w, self._h, self._fps = w, h, 60

    def speak(self, text: str):
        print(f"\n[talker] Processing: \"{text}\"")
        wav, schedule, emotion_events = self._scheduler.build(text)
        print(f"[talker] {len(schedule)} viseme events, {len(emotion_events)} emotion events")
        self._schedule_reader = ScheduleReader(schedule, emotion_events)
        done_cb = None
        if self._auto_exit:
            def done_cb():
                # Schedule exit after a short delay so face returns to neutral
                self._auto_exit_at = time.time() + 0.8
        self.audio.play_wav(wav, done_callback=done_cb)

    def play_file(self, path: str):
        self._schedule_reader = None
        self.audio.play_wav(path)

    def mic_mode(self):
        self._mic_mode = True
        self._schedule_reader = None
        self.audio.start_mic()

    def run(self):
        prev = time.time()
        while self.running:
            now = time.time()
            dt  = min(now - prev, 0.05)
            prev = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_t:
                        threading.Thread(target=self._prompt_speak, daemon=True).start()
                    elif event.key == pygame.K_d:
                        self.debug = not self.debug

            # Auto-exit after speech finishes (for test scripts)
            if self._auto_exit_at and now >= self._auto_exit_at:
                self.running = False
                break

            rms, playback_t = self.audio.get_state()
            if self._schedule_reader and not self._mic_mode:
                viseme = self._schedule_reader.current_viseme(playback_t)
                emotion = self._schedule_reader.current_emotion(playback_t)
            else:
                viseme = amp_to_viseme(rms)
                emotion = Emotion.NEUTRAL
            # Default emotion overrides neutral (when no inline tag active)
            if self._default_emotion and emotion == Emotion.NEUTRAL:
                emotion = self._default_emotion

            self.renderer.update(viseme, dt, emotion)
            self.renderer.draw(self.screen)

            if self.debug:
                self._draw_debug(rms, playback_t, viseme)
            self._draw_hud()

            pygame.display.flip()
            self.clock.tick(self._fps)

        self.audio.close()
        pygame.quit()

    def _prompt_speak(self):
        text = input("\nEnter text to speak: ").strip()
        if text:
            self.speak(text)

    def _draw_hud(self):
        for i, h in enumerate(["T=type & speak", "D=debug", "ESC=quit"]):
            self.screen.blit(
                self.font_sm.render(h, True, (55, 55, 55)), (10, 10 + i * 22)
            )

    def _draw_debug(self, rms, playback_t, viseme):
        emo_label = getattr(self.renderer, 'current_emotion_label', 'neutral')
        lines = [
            f"RMS:     {rms:6.0f}",
            f"Time:    {playback_t:.3f}s",
            f"Viseme:  {viseme.value}",
            f"Shape:   {self.renderer.current_viseme_label}",
            f"Emotion: {emo_label}",
            f"Open:    {self.renderer._open:.2f}",
            f"Width:   {self.renderer._width_t:.2f}",
            f"Round:   {self.renderer._rounded_blend:.2f}",
            f"FPS:     {self.clock.get_fps():.0f}",
        ]
        x = self._w - 240
        for i, line in enumerate(lines):
            self.screen.blit(
                self.font_sm.render(line, True, (0, 220, 80)), (x, 10 + i * 22)
            )


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Talker v2 — phoneme-synced animated face")
    parser.add_argument("--text",     "-t", type=str)
    parser.add_argument("--file",     "-f", type=str)
    parser.add_argument("--mic",      "-m", action="store_true")
    parser.add_argument("--debug",    "-d", action="store_true")
    parser.add_argument("--face",     type=str, default="pumpkin",
                        help="Face name (matches a folder under faces/)")
    parser.add_argument("--face-dir", type=str, default=None,
                        help="Path to a face directory with face.json (overrides --face)")
    parser.add_argument("--emotion",  type=str, default=None,
                        help="Default emotion: neutral, happy, angry, annoyed, sad, surprise")
    parser.add_argument("--auto-exit", action="store_true",
                        help="Exit automatically after speech finishes (for scripts)")
    args = parser.parse_args()

    assets = None
    cfg    = None

    # Init pygame + temporary display so asset loading can use convert_alpha()
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)

    # Resolve face directory: explicit --face-dir, or auto-detect from --face name
    face_dir = args.face_dir
    if not face_dir:
        # Check if a face directory exists for the built-in face name
        candidate = os.path.join(os.path.dirname(__file__) or ".", "faces", args.face)
        if os.path.isfile(os.path.join(candidate, "face.json")):
            face_dir = candidate

    if face_dir:
        try:
            assets = FaceAssetLoader().load(face_dir)
        except Exception as e:
            print(f"[error] Could not load face from {face_dir}: {e}")
            sys.exit(1)
    else:
        cfg = FaceConfig(name=args.face)

    app = TalkerApp(cfg=cfg, assets=assets, debug=args.debug,
                    default_emotion=args.emotion,
                    auto_exit=args.auto_exit)

    if args.text:
        threading.Thread(target=app.speak, args=(args.text,), daemon=True).start()
    elif args.file:
        app.play_file(args.file)
    elif args.mic:
        app.mic_mode()
    else:
        print("No input — press T in window, or use --text / --file / --mic")

    app.run()
