"""
textured_eye.py
===============
Live eyes composed per frame from texture parts, after Adafruit's Uncanny
Eyes: the pupil and iris move inside a fixed outline, the pupil dilates, the
upper lid follows the pupil, blinks close fast and open slow. Everything is
numpy lookups on a small grid, so it runs in real time on a Raspberry Pi.

Parts (a folder, e.g. faces/goat/eye/):
    iris.png        polar strip: x = angle around the pupil, y = distance from
                    the pupil edge outward (Uncanny's convert/<eye>/iris.png)
    pupilMap.png    grey distance field, dark at the pupil centre; encodes
                    slit shapes. Optional: radial if absent.
    lid-upper.png   grey masks: a pixel is visible while its value is above
    lid-lower.png   the lid threshold; raising the threshold closes the lid.
    sclera.png      optional white of the eye (if absent or black, the iris
                    fills the whole eye)
    highlight.png   optional RGBA overlay that stays put (reflections)

EyeMotion holds the state for a pair of eyes (they move together);
TexturedEye renders one eye from that state.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:      # the compositor is testable without a display
    pygame = None

from phoneme_scheduler import Emotion


# ─────────────────────────────────────────────────────
# MOTION MODEL  (timings from uncannyEyes.ino)
# ─────────────────────────────────────────────────────
def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# pupil size targets per emotion, as a fraction added to the base
EMOTION_DILATION = {
    Emotion.NEUTRAL: 0.0, Emotion.HAPPY: 0.05, Emotion.SURPRISE: 0.30,
    Emotion.ANGRY: -0.35, Emotion.ANNOYED: -0.15, Emotion.SAD: -0.05,
}


@dataclass
class EyeMotionConfig:
    gaze_radius: float = 0.35        # how far the pupil wanders, as a fraction of eye radius
    hold_s: Tuple[float, float] = (0.0, 3.0)         # still time between saccades
    move_s: Tuple[float, float] = (0.072, 0.144)     # saccade duration
    micro: float = 0.02              # tiny jitter while holding (fraction of radius)
    blink_close_s: Tuple[float, float] = (0.036, 0.072)
    blink_gap_s: float = 4.0         # next blink: 3x blink length + up to this
    pupil_min: float = 0.12          # pupil size as a fraction of the distance field
    pupil_max: float = 0.40
    pupil_base: float = 0.22
    lid_tracking: float = 0.35       # how much the upper lid follows a downward gaze


class EyeMotion:
    """Shared state for both eyes: gaze offset (fraction of radius), blink 0..1, pupil size."""

    def __init__(self, cfg: Optional[EyeMotionConfig] = None, rng=None, clock=time.monotonic):
        self.cfg = cfg or EyeMotionConfig()
        self.rng = rng or np.random.default_rng()
        self.clock = clock
        now = clock()
        self.gaze = np.zeros(2)
        self._g_from = np.zeros(2)
        self._g_to = np.zeros(2)
        self._g_start = now
        self._g_dur = 0.1
        self._next_move = now + self.rng.uniform(*self.cfg.hold_s)
        self._moving = False
        self.blink = 0.0             # 0 open .. 1 closed
        self._blink_phase = 0        # 0 idle, 1 closing, 2 opening
        self._blink_start = now
        self._blink_dur = 0.05
        self._next_blink = now + 1.0 + self.rng.uniform(0, self.cfg.blink_gap_s)
        self.pupil = self.cfg.pupil_base
        self._pupil_noise = 0.0
        self._pupil_target = 0.0
        self._pupil_next = now
        self.look_target: Optional[np.ndarray] = None   # external look-at (fraction of radius)

    # external control (a camera, a joystick, the voice direction)
    def look_at(self, x: float, y: float) -> None:
        self.look_target = np.array([x, y], dtype=float)

    def release(self) -> None:
        self.look_target = None

    def _pick_target(self) -> np.ndarray:
        if self.look_target is not None:
            return np.clip(self.look_target, -1, 1) * self.cfg.gaze_radius
        # uniform in a disc; 30% of the time return near centre
        if self.rng.random() < 0.3:
            return self.rng.normal(0, 0.05, 2)
        a = self.rng.uniform(0, 2 * math.pi)
        r = math.sqrt(self.rng.random())
        return np.array([math.cos(a), 0.6 * math.sin(a)]) * r * self.cfg.gaze_radius

    def update(self, dt: float, emotion: Emotion = Emotion.NEUTRAL, blink_scale: float = 1.0) -> None:
        now = self.clock()
        c = self.cfg
        # ── saccades ──
        if self._moving:
            t = (now - self._g_start) / self._g_dur
            self.gaze = self._g_from + (self._g_to - self._g_from) * _smoothstep(t)
            if t >= 1.0:
                self._moving = False
                self.gaze = self._g_to.copy()
                self._next_move = now + self.rng.uniform(*c.hold_s)
        elif now >= self._next_move or (self.look_target is not None and
                                        np.linalg.norm(self._g_to - self._pick_target()) > 0.02):
            self._g_from = self.gaze.copy()
            self._g_to = self._pick_target()
            self._g_start = now
            self._g_dur = self.rng.uniform(*c.move_s)
            self._moving = True
        else:
            self.gaze = self._g_to + self.rng.normal(0, c.micro * c.gaze_radius, 2) * 0.5

        # ── blink ──
        if self._blink_phase == 0 and now >= self._next_blink * (1.0 / max(0.1, blink_scale)):
            self._blink_phase = 1
            self._blink_start = now
            self._blink_dur = self.rng.uniform(*c.blink_close_s)
        if self._blink_phase == 1:
            t = (now - self._blink_start) / self._blink_dur
            self.blink = min(1.0, t)
            if t >= 1.0:
                self._blink_phase = 2
                self._blink_start = now
                self._blink_dur *= 2.0                       # open at half speed
        elif self._blink_phase == 2:
            t = (now - self._blink_start) / self._blink_dur
            self.blink = max(0.0, 1.0 - t)
            if t >= 1.0:
                self._blink_phase = 0
                self._next_blink = now + self._blink_dur * 1.5 + self.rng.uniform(0, c.blink_gap_s)
        # ── pupil: slow wander + emotion ──
        if now >= self._pupil_next:
            self._pupil_target = self.rng.uniform(-0.06, 0.06)
            self._pupil_next = now + self.rng.uniform(0.4, 2.5)
        self._pupil_noise += (self._pupil_target - self._pupil_noise) * min(1.0, 2.0 * dt)
        want = c.pupil_base * (1.0 + EMOTION_DILATION.get(emotion, 0.0)) + self._pupil_noise * c.pupil_base
        want = max(c.pupil_min, min(c.pupil_max, want))
        self.pupil += (want - self.pupil) * min(1.0, 6.0 * dt)

    @property
    def upper_lid_extra(self) -> float:
        """Extra closure of the upper lid from looking down (0..lid_tracking)."""
        return self.cfg.lid_tracking * max(0.0, float(self.gaze[1])) / max(1e-6, self.cfg.gaze_radius)


# ─────────────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────────────
def _gray(path: str, size: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


class TexturedEyeAssets:
    def __init__(self, folder: str, size: int = 224, mirror: bool = False):
        from PIL import Image
        self.size = size
        self.mirror = mirror
        iris = Image.open(os.path.join(folder, "iris.png")).convert("RGB")
        self.iris = np.asarray(iris, dtype=np.float32) / 255.0            # (h, w, 3)
        if mirror:
            self.iris = self.iris[:, ::-1]
        p = os.path.join(folder, "pupilMap.png")
        self.pupil_map = _gray(p, size) if os.path.exists(p) else None
        self.lid_upper = _gray(os.path.join(folder, "lid-upper.png"), size) if os.path.exists(os.path.join(folder, "lid-upper.png")) else None
        self.lid_lower = _gray(os.path.join(folder, "lid-lower.png"), size) if os.path.exists(os.path.join(folder, "lid-lower.png")) else None
        if mirror:
            if self.pupil_map is not None:
                self.pupil_map = self.pupil_map[:, ::-1]
            if self.lid_upper is not None:
                self.lid_upper = self.lid_upper[:, ::-1]
            if self.lid_lower is not None:
                self.lid_lower = self.lid_lower[:, ::-1]
        self.sclera = None
        sp = os.path.join(folder, "sclera.png")
        if os.path.exists(sp):
            sc = np.asarray(Image.open(sp).convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
            if sc.max() > 0.05:
                self.sclera = sc[:, ::-1] if mirror else sc
        self.highlight = None
        hp = os.path.join(folder, "highlight.png")
        if os.path.exists(hp):
            hl = Image.open(hp).convert("RGBA").resize((size, size), Image.BILINEAR)
            if mirror:
                hl = hl.transpose(Image.FLIP_LEFT_RIGHT)
            self.highlight = np.asarray(hl, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────
# COMPOSITOR
# ─────────────────────────────────────────────────────
class TexturedEye:
    """
    Renders one eye. The pupil/iris origin shifts with the gaze inside a
    fixed outline; the distance field (pupil map) moves with it, so slit
    pupils keep their shape. Precomputes an oversized polar grid and slices
    a window per frame instead of recomputing atan2.
    """

    def __init__(self, assets: TexturedEyeAssets, lid_open: float = 0.55, iris_radius: float = 0.62):
        self.a = assets
        n = assets.size
        self.n = n
        self.lid_open = lid_open
        self.iris_radius = iris_radius
        self.max_shift = int(n * 0.45)
        big = n + 2 * self.max_shift
        yy, xx = np.mgrid[0:big, 0:big].astype(np.float32)
        c = (big - 1) / 2
        dx, dy = xx - c, yy - c
        self._ang_big = ((np.arctan2(dy, dx) + math.pi) / (2 * math.pi)).astype(np.float32)
        r_big = np.sqrt(dx * dx + dy * dy) / (n / 2)
        if assets.pupil_map is not None:
            # place the distance field on the big canvas (it is defined on the eye disc)
            pm = np.ones((big, big), dtype=np.float32)
            pm[self.max_shift:self.max_shift + n, self.max_shift:self.max_shift + n] = assets.pupil_map
            self._dist_big = np.maximum(pm, np.where(r_big > 1.0, 1.0, 0.0))
        else:
            self._dist_big = np.clip(r_big, 0, 1).astype(np.float32)
        self._r_big = r_big.astype(np.float32)
        # fixed outline (eye disc) and rim darkening on the un-shifted grid
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        cc = (n - 1) / 2
        self._r = np.sqrt((xx - cc) ** 2 + (yy - cc) ** 2) / (n / 2)
        self._disc = (self._r <= 1.0).astype(np.float32)
        self._rim = (0.35 + 0.65 * np.clip((1.0 - self._r) / 0.12, 0, 1)).astype(np.float32)
        self._last_key = None
        self._last_frame: Optional[np.ndarray] = None

    def render(self, gaze_frac: Tuple[float, float], pupil: float, blink: float,
               lid_extra: float = 0.0) -> np.ndarray:
        """Returns an (n, n, 4) uint8 RGBA array."""
        n = self.n
        gx = int(round(float(gaze_frac[0]) * n * 0.5))
        gy = int(round(float(gaze_frac[1]) * n * 0.5))
        gx = max(-self.max_shift, min(self.max_shift, gx))
        gy = max(-self.max_shift, min(self.max_shift, gy))
        pupil_q = round(pupil, 3)
        blink_q = round(blink, 2)
        key = (gx, gy, pupil_q, blink_q, round(lid_extra, 2))
        if key == self._last_key and self._last_frame is not None:
            return self._last_frame

        # window of the big polar grid shifted by the gaze
        y0 = self.max_shift - gy
        x0 = self.max_shift - gx
        ang = self._ang_big[y0:y0 + n, x0:x0 + n]
        dist = self._dist_big[y0:y0 + n, x0:x0 + n]
        rr = self._r_big[y0:y0 + n, x0:x0 + n]

        ih, iw, _ = self.a.iris.shape
        d = np.clip((dist - pupil_q) / max(1e-6, 1.0 - pupil_q), 0, 0.999)
        iy = (d * (ih - 1)).astype(np.int32)
        ix = (ang * (iw - 1)).astype(np.int32)
        rgb = self.a.iris[iy, ix]
        edge = np.clip((dist - pupil_q) / 0.03, 0, 1)[..., None]
        rgb = rgb * edge                                    # pupil is black with a soft edge

        if self.a.sclera is not None:
            in_iris = (rr <= self.iris_radius)[..., None]
            limbus = np.clip((self.iris_radius - rr) / 0.05, 0, 1)[..., None]
            rgb = np.where(in_iris, rgb * (0.55 + 0.45 * limbus), self.a.sclera)

        rgb = rgb * self._rim[..., None]

        # lids: threshold rises with blink and with looking down
        u_thr = self.lid_open + (1.0 - self.lid_open) * blink_q + lid_extra * (1.0 - blink_q)
        l_thr = self.lid_open + (1.0 - self.lid_open) * blink_q * 0.6
        alpha = self._disc.copy()
        if self.a.lid_upper is not None:
            alpha *= np.clip((self.a.lid_upper - u_thr) / 0.04 + 0.5, 0, 1)
        if self.a.lid_lower is not None:
            alpha *= np.clip((self.a.lid_lower - l_thr) / 0.04 + 0.5, 0, 1)

        if self.a.highlight is not None:
            ha = self.a.highlight[..., 3:4]
            rgb = rgb * (1 - ha) + self.a.highlight[..., :3] * ha

        out = np.empty((n, n, 4), dtype=np.uint8)
        out[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        out[..., 3] = (alpha * 255).astype(np.uint8)
        self._last_key = key
        self._last_frame = out
        return out

    def surface(self, *args, **kwargs):
        """pygame Surface of render(*args, **kwargs)."""
        frame = self.render(*args, **kwargs)
        surf = pygame.image.frombuffer(np.ascontiguousarray(frame).tobytes(), (self.n, self.n), "RGBA")
        return surf.convert_alpha() if pygame.display.get_surface() is not None else surf
