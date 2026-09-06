"""
face_asset_loader.py
====================
Loads artist-created face assets (PNG) from a face directory and renders them.

Directory structure:
  faces/<face_name>/
      face.json          <- manifest (required)
      face_base.png      <- full face background art (optional)
      eye_left.png       <- left eye art (optional)
      eye_right.png      <- right eye art (optional)
      nose.png           <- nose art (optional)
      mouth_<key>.png    <- mouth state images (optional, any subset)

All art files are optional. Missing files fall back to procedural drawing.
Everything that can be precomputed (scaling, opacity, glow sprites) is done
once at load time; per-frame work is blits plus a few small polygons.
"""

from __future__ import annotations

import collections
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from phoneme_scheduler import Emotion, VISEME_PROPS, Viseme


# ─────────────────────────────────────────────────────────
# FACE MANIFEST  (face.json schema)
# ─────────────────────────────────────────────────────────
@dataclass
class EyeConfig:
    image:   Optional[str] = None   # filename relative to face dir
    cx:      int = 0
    cy:      int = 0
    scale:   float = 1.0
    opacity: float = 1.0            # 0.0=invisible, 1.0=fully opaque


@dataclass
class MouthConfig:
    anchor_cx: int = 400
    anchor_cy: int = 540
    max_w:     int = 260
    min_w:     int = 160
    scale:     float = 1.0          # scale mouth PNGs around the anchor point
    offset_x:  int = 0              # shift mouth PNGs without moving the anchor
    offset_y:  int = 0
    opacity:   float = 1.0
    # Procedural mouth (used when no mouth images are provided)
    color:      Tuple = (255, 200, 0)
    dark_color: Tuple = (10, 10, 10)
    style:      str = "toothed"     # "toothed" | "rounded"
    n_teeth:    int = 5


@dataclass
class NoseConfig:
    image:   Optional[str] = None
    cx:      int = 400
    cy:      int = 440
    scale:   float = 1.0
    opacity: float = 1.0


@dataclass
class FaceManifest:
    name:         str = "custom"
    description:  str = ""
    canvas_w:     int = 800
    canvas_h:     int = 800
    fps:          int = 60
    bg_color:     Tuple = (0, 0, 0)

    face_base:         Optional[str] = None
    face_base_opacity: float = 1.0
    face_color:        Tuple = (210, 100, 0)
    face_outline:      Tuple = (140, 60, 0)
    glow_color:        Tuple = (255, 160, 0)
    glow_intensity:    float = 1.0      # 0=no glow, 1=normal, 2=intense

    eye_left:  EyeConfig = field(default_factory=EyeConfig)
    eye_right: EyeConfig = field(default_factory=EyeConfig)
    eye_color: Tuple = (255, 200, 0)    # procedural eyes
    blink:     bool = True
    eye_speech_pulse: float = 0.0       # eyes grow by this fraction when the mouth is fully open
    blink_interval: Tuple = (2.5, 5.5)  # seconds between blinks (random in range; emotion scales it)
    blink_speed: float = 9.0            # how fast a blink closes/opens (higher = snappier)
    # Idle glances: eyes drift to a random point within `amount` px every `interval`
    # seconds while idle; `speed` is the easing rate; `while_speaking` scales the
    # movement during speech (0 = eyes lock forward when talking).
    gaze_amount: float = 0.0
    gaze_interval: Tuple = (2.0, 6.0)
    gaze_speed: float = 2.5
    gaze_while_speaking: float = 0.3

    draw_nose:  bool = False
    nose_color: Tuple = (255, 200, 0)
    nose:       NoseConfig = field(default_factory=NoseConfig)

    mouth:        MouthConfig = field(default_factory=MouthConfig)
    mouth_images: Dict[str, str] = field(default_factory=dict)   # viseme key -> filename

    draw_stem:  bool = False            # pumpkin-type faces
    stem_color: Tuple = (60, 120, 20)

    # Defaults for the voice loop (CLI flags override). voices is keyed by TTS
    # backend name: {"elevenlabs": "<voice id>", "edge": "en-US-AriaNeural"};
    # tts_model picks the ElevenLabs model (e.g. "eleven_v3").
    voices:     Dict[str, str] = field(default_factory=dict)
    tts_model:  Optional[str] = None
    character:  str = ""                # persona line for the LLM

    _KNOWN_TOP = {
        "name", "description", "canvas_w", "canvas_h", "fps", "bg_color", "face_base",
        "face_base_opacity", "face_color", "face_outline", "glow_color", "glow_intensity",
        "eye_left", "eye_right", "eye_color", "blink", "blink_interval", "blink_speed",
        "eye_speech_pulse", "gaze", "draw_nose", "nose_color", "nose",
        "mouth", "mouth_images", "draw_stem", "stem_color", "voices", "tts_model", "character",
    }
    _KNOWN_EYE = {"image", "cx", "cy", "scale", "opacity"}
    _KNOWN_MOUTH = {"anchor_cx", "anchor_cy", "max_w", "min_w", "scale", "offset_x", "offset_y",
                    "opacity", "color", "dark_color", "style", "n_teeth"}
    _KNOWN_NOSE = {"image", "cx", "cy", "scale", "opacity"}

    @staticmethod
    def _warn_unknown(section: str, d: dict, known: set) -> None:
        for k in d:
            if not k.startswith("_") and k not in known:
                print(f"[assets] warning: unknown key {k!r} in {section} (typo?)")

    @staticmethod
    def from_dict(d: dict) -> "FaceManifest":
        m = FaceManifest()
        FaceManifest._warn_unknown("face.json", d, FaceManifest._KNOWN_TOP)
        m.name        = d.get("name", m.name)
        m.description = d.get("description", m.description)
        m.canvas_w    = int(d.get("canvas_w", m.canvas_w))
        m.canvas_h    = int(d.get("canvas_h", m.canvas_h))
        m.fps         = int(d.get("fps", m.fps))
        m.bg_color    = tuple(d.get("bg_color", m.bg_color))
        m.face_base   = d.get("face_base")
        m.face_base_opacity = float(d.get("face_base_opacity", m.face_base_opacity))
        m.face_color  = tuple(d.get("face_color", m.face_color))
        m.face_outline = tuple(d.get("face_outline", m.face_outline))
        m.glow_color  = tuple(d.get("glow_color", m.glow_color))
        m.glow_intensity = float(d.get("glow_intensity", m.glow_intensity))
        m.eye_color   = tuple(d.get("eye_color", m.eye_color))
        m.blink       = bool(d.get("blink", m.blink))
        m.eye_speech_pulse = float(d.get("eye_speech_pulse", m.eye_speech_pulse))
        m.blink_interval = tuple(float(x) for x in d.get("blink_interval", m.blink_interval))
        m.blink_speed = float(d.get("blink_speed", m.blink_speed))
        gz = d.get("gaze", {})
        FaceManifest._warn_unknown("gaze", gz, {"amount", "interval", "speed", "while_speaking"})
        m.gaze_amount = float(gz.get("amount", m.gaze_amount))
        m.gaze_interval = tuple(float(x) for x in gz.get("interval", m.gaze_interval))
        m.gaze_speed = float(gz.get("speed", m.gaze_speed))
        m.gaze_while_speaking = float(gz.get("while_speaking", m.gaze_while_speaking))
        m.draw_nose   = bool(d.get("draw_nose", m.draw_nose))
        m.nose_color  = tuple(d.get("nose_color", m.nose_color))
        m.draw_stem   = bool(d.get("draw_stem", m.draw_stem))
        m.stem_color  = tuple(d.get("stem_color", m.stem_color))
        m.mouth_images = {str(k).lower(): v for k, v in d.get("mouth_images", {}).items()}
        m.voices      = {str(k).lower(): str(v) for k, v in d.get("voices", {}).items()}
        m.tts_model   = d.get("tts_model") or None
        m.character   = str(d.get("character", ""))

        for side in ("eye_left", "eye_right"):
            ed = d.get(side, {})
            FaceManifest._warn_unknown(side, ed, FaceManifest._KNOWN_EYE)
            ec = EyeConfig(
                image=ed.get("image"),
                cx=int(ed.get("cx", 280 if side == "eye_left" else 520)),
                cy=int(ed.get("cy", 320)),
                scale=float(ed.get("scale", 1.0)),
                opacity=float(ed.get("opacity", 1.0)),
            )
            setattr(m, side, ec)

        md = d.get("mouth", {})
        FaceManifest._warn_unknown("mouth", md, FaceManifest._KNOWN_MOUTH)
        mc = MouthConfig()
        mc.anchor_cx  = int(md.get("anchor_cx", mc.anchor_cx))
        mc.anchor_cy  = int(md.get("anchor_cy", mc.anchor_cy))
        mc.max_w      = int(md.get("max_w", mc.max_w))
        mc.min_w      = int(md.get("min_w", mc.min_w))
        mc.scale      = float(md.get("scale", mc.scale))
        mc.offset_x   = int(md.get("offset_x", mc.offset_x))
        mc.offset_y   = int(md.get("offset_y", mc.offset_y))
        mc.opacity    = float(md.get("opacity", mc.opacity))
        mc.color      = tuple(md.get("color", mc.color))
        mc.dark_color = tuple(md.get("dark_color", mc.dark_color))
        mc.style      = str(md.get("style", mc.style))
        mc.n_teeth    = int(md.get("n_teeth", mc.n_teeth))
        m.mouth = mc

        nd = d.get("nose", {})
        FaceManifest._warn_unknown("nose", nd, FaceManifest._KNOWN_NOSE)
        m.nose = NoseConfig(
            image=nd.get("image"),
            cx=int(nd.get("cx", 400)), cy=int(nd.get("cy", 440)),
            scale=float(nd.get("scale", 1.0)), opacity=float(nd.get("opacity", 1.0)),
        )
        return m


def default_manifest(name: str = "default") -> FaceManifest:
    """A fully procedural face for names that have no faces/<name>/ directory."""
    m = FaceManifest()
    m.name = name
    m.description = "Procedural fallback face"
    m.eye_left = EyeConfig(cx=280, cy=320)
    m.eye_right = EyeConfig(cx=520, cy=320)
    return m


# ─────────────────────────────────────────────────────────
# EMOTION PARAMETERS  (eye modulation per emotion)
# ─────────────────────────────────────────────────────────
# eye_squish: vertical scale (<1 = narrow/squint, >1 = wide)
# eye_tilt:   degrees rotation, positive = inner corners down (angry)
# blink_mult: multiplier on blink frequency
# speed:      lerp rate for the transition (higher = snappier)
EMOTION_PARAMS = {
    Emotion.NEUTRAL:  {"eye_squish": 1.0,  "eye_tilt": 0.0,  "blink_mult": 1.0, "speed": 4.0},
    Emotion.HAPPY:    {"eye_squish": 0.85, "eye_tilt": 0.0,  "blink_mult": 1.2, "speed": 5.0},
    Emotion.ANGRY:    {"eye_squish": 0.6,  "eye_tilt": 12.0, "blink_mult": 0.7, "speed": 8.0},
    Emotion.ANNOYED:  {"eye_squish": 0.7,  "eye_tilt": 5.0,  "blink_mult": 0.8, "speed": 5.0},
    Emotion.SAD:      {"eye_squish": 0.85, "eye_tilt": -8.0, "blink_mult": 1.5, "speed": 3.0},
    Emotion.SURPRISE: {"eye_squish": 1.3,  "eye_tilt": 0.0,  "blink_mult": 0.3, "speed": 10.0},
}


# ─────────────────────────────────────────────────────────
# LOADED FACE ASSETS  (pygame Surfaces, ready to blit)
# ─────────────────────────────────────────────────────────
VISEME_TO_MOUTH_KEY = {
    Viseme.SIL: ["sil", "closed"],
    Viseme.PP:  ["pp",  "closed", "sil"],
    Viseme.FF:  ["ff",  "ah",     "sil"],
    Viseme.TH:  ["th",  "dd",     "ah"],
    Viseme.DD:  ["dd",  "ah",     "sil"],
    Viseme.KK:  ["kk",  "ah",     "aa"],
    Viseme.CH:  ["ch",  "oo",     "ah"],
    Viseme.SS:  ["ss",  "ee",     "ah"],
    Viseme.AA:  ["aa",  "ah",     "wide"],
    Viseme.EE:  ["ee",  "ah",     "wide"],
    Viseme.OO:  ["oo",  "round",  "ch"],
    Viseme.AH:  ["ah",  "neutral", "sil"],
}


@dataclass
class LoadedFaceAssets:
    manifest:    FaceManifest
    face_base:   Optional[pygame.Surface] = None
    eye_left:    Optional[pygame.Surface] = None
    eye_right:   Optional[pygame.Surface] = None
    nose:        Optional[pygame.Surface] = None
    # Art is cropped to its opaque bounds at load; these offsets restore placement.
    eye_left_offset:  Tuple[int, int] = (0, 0)
    eye_right_offset: Tuple[int, int] = (0, 0)
    nose_offset:      Tuple[int, int] = (0, 0)
    # Mouth surfaces are already scaled + alpha'd + cropped; mouth_pos[key] is the blit origin.
    mouth_surfs: Dict[str, pygame.Surface] = field(default_factory=dict)
    mouth_pos:   Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Resolved once: viseme -> (surface, blit_pos) or None
    mouth_for_viseme: Dict[Viseme, Optional[Tuple[pygame.Surface, Tuple[int, int]]]] = field(default_factory=dict)

    def get_mouth(self, viseme: Viseme) -> Optional[Tuple[pygame.Surface, Tuple[int, int]]]:
        return self.mouth_for_viseme.get(viseme)

    def has_mouth_art(self) -> bool:
        return bool(self.mouth_surfs)

    def resolve_mouths(self) -> None:
        for v in Viseme:
            found = None
            for key in VISEME_TO_MOUTH_KEY.get(v, []):
                if key in self.mouth_surfs:
                    found = key
                    break
            if found is None and self.mouth_surfs:
                found = next(iter(self.mouth_surfs))
            self.mouth_for_viseme[v] = None if found is None else \
                (self.mouth_surfs[found], self.mouth_pos[found])


def _load_image(path: str) -> Optional[pygame.Surface]:
    if not os.path.exists(path):
        return None
    try:
        surf = pygame.image.load(path)
        try:
            return surf.convert_alpha()
        except pygame.error:
            return surf   # no display yet; still usable, just slower to blit
    except Exception as e:
        print(f"[assets] PNG load error for {path}: {e}")
        return None


def _scaled(surf: pygame.Surface, scale: float) -> pygame.Surface:
    if scale == 1.0:
        return surf
    w = max(1, int(surf.get_width() * scale))
    h = max(1, int(surf.get_height() * scale))
    return pygame.transform.smoothscale(surf, (w, h))


def _cropped(surf: pygame.Surface) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """
    Trim fully transparent borders. Returns (cropped, offset_of_crop_center
    relative_to_original_center). Artists often export the whole artboard;
    transforming and blitting only the opaque part is several times cheaper.
    """
    rect = surf.get_bounding_rect()
    if rect.size == surf.get_size() or rect.width == 0 or rect.height == 0:
        return surf, (0, 0)
    cropped = surf.subsurface(rect).copy()
    dx = rect.centerx - surf.get_width() // 2
    dy = rect.centery - surf.get_height() // 2
    return cropped, (dx, dy)


def _with_opacity(surf: pygame.Surface, opacity: float) -> pygame.Surface:
    if opacity < 1.0:
        surf.set_alpha(int(max(0.0, opacity) * 255))
    return surf


# ─────────────────────────────────────────────────────────
# FACE ASSET LOADER
# ─────────────────────────────────────────────────────────
class FaceAssetLoader:
    """Loads face.json + art from a directory. Needs a pygame display for convert_alpha()."""

    def load(self, face_dir: str) -> LoadedFaceAssets:
        manifest_path = os.path.join(face_dir, "face.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No face.json found in {face_dir}")
        with open(manifest_path) as f:
            manifest = FaceManifest.from_dict(json.load(f))
        return self.build(manifest, face_dir)

    def build(self, manifest: FaceManifest, face_dir: Optional[str] = None) -> LoadedFaceAssets:
        assets = LoadedFaceAssets(manifest=manifest)
        print(f"[assets] Loading face: {manifest.name}")

        def path_of(name: str) -> str:
            return os.path.join(face_dir or ".", name)

        if manifest.face_base:
            surf = _load_image(path_of(manifest.face_base))
            if surf:
                assets.face_base = _with_opacity(surf, manifest.face_base_opacity)
                print(f"[assets]   face_base: {manifest.face_base} OK")
            else:
                print(f"[assets]   face_base: {manifest.face_base} not found, skipping")

        for side in ("eye_left", "eye_right"):
            ec: EyeConfig = getattr(manifest, side)
            if ec.image:
                surf = _load_image(path_of(ec.image))
                if surf:
                    surf, (dx, dy) = _cropped(_scaled(surf, ec.scale))
                    setattr(assets, side, surf)
                    setattr(assets, side + "_offset", (dx, dy))
                    print(f"[assets]   {side}: {ec.image} OK ({surf.get_width()}x{surf.get_height()} after crop)")
                else:
                    print(f"[assets]   {side}: {ec.image} not found, procedural")

        nc = manifest.nose
        if nc.image:
            surf = _load_image(path_of(nc.image))
            if surf:
                surf, assets.nose_offset = _cropped(_scaled(surf, nc.scale))
                assets.nose = _with_opacity(surf, nc.opacity)
                print(f"[assets]   nose: {nc.image} OK")

        mc = manifest.mouth
        base_pos = self._mouth_blit_pos(mc)
        for key, filename in manifest.mouth_images.items():
            surf = _load_image(path_of(filename))
            if surf:
                scaled = _scaled(surf, mc.scale)
                rect = scaled.get_bounding_rect()
                if rect.width and rect.height and rect.size != scaled.get_size():
                    scaled = scaled.subsurface(rect).copy()
                assets.mouth_surfs[key] = _with_opacity(scaled, mc.opacity)
                assets.mouth_pos[key] = (base_pos[0] + rect.x, base_pos[1] + rect.y)
                print(f"[assets]   mouth/{key}: {filename} OK")
            else:
                print(f"[assets]   mouth/{key}: {filename} not found")
        assets.resolve_mouths()

        print(f"[assets] Loaded {manifest.name} — "
              f"base={'art' if assets.face_base else 'none'}, "
              f"eyes={'art' if assets.eye_left else 'procedural'}, "
              f"nose={'art' if assets.nose else 'none'}, "
              f"mouth={'art' if assets.has_mouth_art() else 'procedural'}")
        return assets

    @staticmethod
    def _mouth_blit_pos(mc: MouthConfig) -> Tuple[int, int]:
        """Full-canvas mouth PNGs scale around the anchor point, then shift by offset."""
        bx, by = mc.offset_x, mc.offset_y
        if mc.scale != 1.0:
            bx += int(mc.anchor_cx - mc.anchor_cx * mc.scale)
            by += int(mc.anchor_cy - mc.anchor_cy * mc.scale)
        return (bx, by)


# ─────────────────────────────────────────────────────────
# GLOW HELPERS  (draw into small surfaces, not the whole canvas)
# ─────────────────────────────────────────────────────────
def _polygon_glow(pts: List[Tuple[int, int]], color, layers: int, spread: float
                  ) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """Render an expanding-polygon halo into a surface just big enough for it."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    max_scale = 1.0 + spread / 60.0
    xs = [cx + (x - cx) * max_scale for x, _ in pts]
    ys = [cy + (y - cy) * max_scale for _, y in pts]
    x0, y0 = int(min(xs)) - 2, int(min(ys)) - 2
    w = int(max(xs)) - x0 + 4
    h = int(max(ys)) - y0 + 4
    gs = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        scale = 1.0 + (i / layers) * (spread / 60.0)
        alpha = int(80 * (i / layers) ** 1.4)
        expanded = [(int(cx + (x - cx) * scale) - x0, int(cy + (y - cy) * scale) - y0)
                    for x, y in pts]
        pygame.draw.polygon(gs, (*color, alpha), expanded)
    return gs, (x0, y0)


def _shadow_sprite(color, radius: int = 44) -> pygame.Surface:
    """Soft radial blob used as a drop shadow; position-independent, so cached."""
    size = radius * 2 + 2
    gs = pygame.Surface((size, size), pygame.SRCALPHA)
    c = (radius + 1, radius + 1)
    for r in range(radius, 0, -8):
        alpha = int(120 * (1 - r / radius) ** 1.5)
        pygame.draw.circle(gs, (*color, alpha), c, r)
    return gs


# ─────────────────────────────────────────────────────────
# ASSET-AWARE FACE RENDERER
# ─────────────────────────────────────────────────────────
class AssetFaceRenderer:
    """
    Renders a face from loaded assets, falling back to procedural shapes for
    missing pieces. Handles all 12 visemes + blink + emotion + glow.
    """

    EYE_CACHE_SIZE = 96      # squish/tilt variants kept per renderer

    def __init__(self, assets: LoadedFaceAssets):
        self.assets = assets
        self.manifest = assets.manifest
        m = self.manifest

        # Smoothed mouth state
        self._open = 0.0
        self._width_t = 0.8
        self._rounded_blend = 0.0
        self.current_viseme = Viseme.SIL
        self.current_viseme_label = "sil"
        self._mouth: Optional[Tuple[pygame.Surface, Tuple[int, int]]] = assets.get_mouth(Viseme.SIL)

        # Blink
        self._blink_t = 0.0
        self._blink_next = time.monotonic() + np.random.uniform(*m.blink_interval)

        # Idle gaze (pixel offset applied to both eyes)
        self._gaze = np.zeros(2)
        self._gaze_target = np.zeros(2)
        self._gaze_next = time.monotonic() + np.random.uniform(*m.gaze_interval)
        self._last_speech = 0.0

        # Emotion (smoothed toward targets)
        self._emotion = Emotion.NEUTRAL
        self._eye_squish = 1.0
        self._eye_tilt = 0.0
        self._blink_mult = 1.0
        self.current_emotion_label = "neutral"

        # Caches
        self._eye_cache: "collections.OrderedDict[tuple, pygame.Surface]" = collections.OrderedDict()
        self._shadow = _shadow_sprite(m.glow_color)
        self._ambient: Optional[pygame.Surface] = (
            self._build_ambient_glow() if assets.face_base and m.glow_intensity > 0 else None)

    # ── Public update ────────────────────────────────────
    def update(self, viseme: Viseme, dt: float, emotion: Optional[Emotion] = None) -> None:
        props = VISEME_PROPS[viseme]
        self.current_viseme_label = props.label
        if viseme != self.current_viseme:
            self.current_viseme = viseme
            new_mouth = self.assets.get_mouth(viseme)
            if new_mouth is not None:
                self._mouth = new_mouth

        spd_open = 14.0 if props.open_amount > self._open else 7.0
        spd_width = 11.0 if props.width_scale > self._width_t else 6.0
        self._open += (props.open_amount - self._open) * spd_open * dt
        self._width_t += (props.width_scale - self._width_t) * spd_width * dt
        self._open = max(0.0, min(1.0, self._open))
        self._width_t = max(0.0, min(1.0, self._width_t))
        target_round = 1.0 if props.rounded else 0.0
        self._rounded_blend += (target_round - self._rounded_blend) * 9.0 * dt

        if emotion is not None:
            self._emotion = emotion
            self.current_emotion_label = emotion.value
        ep = EMOTION_PARAMS.get(self._emotion, EMOTION_PARAMS[Emotion.NEUTRAL])
        k = ep["speed"] * dt
        self._eye_squish += (ep["eye_squish"] - self._eye_squish) * k
        self._eye_tilt += (ep["eye_tilt"] - self._eye_tilt) * k
        self._blink_mult += (ep["blink_mult"] - self._blink_mult) * k

        now = time.monotonic()
        m = self.manifest
        if m.blink and now >= self._blink_next:
            self._blink_t = 1.0
            self._blink_next = now + np.random.uniform(*m.blink_interval) / max(0.1, self._blink_mult)
        if self._blink_t > 0:
            self._blink_t = max(0.0, self._blink_t - dt * m.blink_speed)

        # Idle glances
        if viseme is not Viseme.SIL:
            self._last_speech = now
        if m.gaze_amount > 0:
            speaking = now - self._last_speech < 0.8
            if now >= self._gaze_next:
                self._gaze_next = now + np.random.uniform(*m.gaze_interval)
                r = np.random.uniform(0.3, 1.0) * m.gaze_amount
                a = np.random.uniform(0, 2 * math.pi)
                # mostly sideways, a little up/down; sometimes back to centre
                self._gaze_target = (np.zeros(2) if np.random.random() < 0.3
                                     else np.array([r * math.cos(a), 0.5 * r * math.sin(a)]))
            target = self._gaze_target * (m.gaze_while_speaking if speaking else 1.0)
            self._gaze += (target - self._gaze) * min(1.0, m.gaze_speed * dt)

    # ── Draw frame ───────────────────────────────────────
    def draw(self, surf: pygame.Surface) -> None:
        m = self.manifest
        surf.fill(m.bg_color)
        if self.assets.face_base:
            if self._ambient is not None:
                self._ambient.set_alpha(int(255 * (35 + 40 * self._open) / 75))
                surf.blit(self._ambient, (0, 0))
            surf.blit(self.assets.face_base, (0, 0))
            if m.draw_stem:
                self._draw_stem(surf)
        self._draw_eye_side(surf, m.eye_left, self.assets.eye_left, self.assets.eye_left_offset, True)
        self._draw_eye_side(surf, m.eye_right, self.assets.eye_right, self.assets.eye_right_offset, False)
        self._draw_nose(surf)
        if self.assets.has_mouth_art():
            if self._mouth is not None:
                surf.blit(self._mouth[0], self._mouth[1])
        elif m.mouth.opacity > 0:            # opacity 0 = no mouth at all (e.g. EVE)
            self._draw_procedural_mouth(surf)

    # ── Glow primitives ──────────────────────────────────
    def _draw_shape_glow(self, surf, pts, layers=6, spread=14) -> None:
        if self.manifest.glow_intensity <= 0:
            return
        gs, pos = _polygon_glow(pts, self.manifest.glow_color, layers, spread)
        surf.blit(gs, pos)

    def _draw_drop_shadow(self, surf, corner) -> None:
        r = self._shadow.get_width() // 2
        surf.blit(self._shadow, (corner[0] - r, corner[1] - r))

    def _build_ambient_glow(self) -> pygame.Surface:
        """Halo behind body art, rendered once at full brightness; alpha-modulated per frame."""
        m = self.manifest
        gs = pygame.Surface((m.canvas_w, m.canvas_h), pygame.SRCALPHA)
        base = int(75 * m.glow_intensity)
        cx, cy = m.canvas_w // 2, m.canvas_h // 2
        rx, ry = int(m.canvas_w * 0.42), int(m.canvas_h * 0.46)
        for r in range(110, 0, -12):
            alpha = max(0, min(255, base - r // 3))
            pygame.draw.ellipse(gs, (*m.glow_color, alpha),
                                (cx - rx - r, cy - ry - r, (rx + r) * 2, (ry + r) * 2))
        return gs

    # ── Stem ─────────────────────────────────────────────
    def _draw_stem(self, surf) -> None:
        m = self.manifest
        cx = m.canvas_w // 2
        top = m.canvas_h // 2 - int(m.canvas_h * 0.44)
        pts = [(cx - 14, top + 10), (cx - 4, top - 58), (cx + 26, top - 78),
               (cx + 30, top - 18), (cx + 14, top + 10)]
        pygame.draw.polygon(surf, m.stem_color, pts)
        pygame.draw.polygon(surf, (40, 90, 10), pts, 2)

    # ── Eyes ─────────────────────────────────────────────
    def _draw_eye_side(self, surf, ec: EyeConfig, eye_surf: Optional[pygame.Surface],
                       offset: Tuple[int, int], is_left: bool) -> None:
        if eye_surf is not None:
            self._draw_art_eye(surf, ec, eye_surf, offset, is_left)
        else:
            self._draw_procedural_eye(surf, ec.cx, ec.cy, is_left)

    def _eye_variant(self, eye_surf: pygame.Surface, new_h: int, tilt: float,
                     opacity: float) -> pygame.Surface:
        """Squished + tilted copy of an eye, cached by quantized parameters."""
        # Quantize: 2 px of squish and 1 degree of tilt are invisible at 60 fps
        new_h = max(2, (new_h // 2) * 2)
        tilt = float(round(tilt))
        key = (id(eye_surf), new_h, tilt)
        cached = self._eye_cache.get(key)
        if cached is not None:
            self._eye_cache.move_to_end(key)
            return cached
        ow, oh = eye_surf.get_size()
        out = eye_surf if new_h == oh else pygame.transform.scale(eye_surf, (ow, new_h))
        if abs(tilt) > 0.5:
            out = pygame.transform.rotate(out, tilt)
        if out is eye_surf and opacity < 1.0:
            out = eye_surf.copy()
        if opacity < 1.0:
            out.set_alpha(int(opacity * 255))
        self._eye_cache[key] = out
        if len(self._eye_cache) > self.EYE_CACHE_SIZE:
            self._eye_cache.popitem(last=False)
        return out

    def _draw_art_eye(self, surf, ec: EyeConfig, eye_surf: pygame.Surface,
                      offset: Tuple[int, int], is_left: bool) -> None:
        blink_squish = max(0.03, 1.0 - self._blink_t * 0.97)
        pulse = 1.0 + self.manifest.eye_speech_pulse * self._open
        new_h = max(2, int(eye_surf.get_height() * blink_squish * self._eye_squish * pulse))
        tilt = -self._eye_tilt if is_left else self._eye_tilt   # + = inner corners down
        img = self._eye_variant(eye_surf, new_h, tilt, ec.opacity)
        cx = ec.cx + offset[0] + int(round(self._gaze[0]))
        cy = ec.cy + offset[1] + int(round(self._gaze[1]))
        surf.blit(img, (cx - img.get_width() // 2, cy - img.get_height() // 2))

    def _draw_procedural_eye(self, surf, cx, cy, is_left: bool) -> None:
        m = self.manifest
        cx += int(round(self._gaze[0]))
        cy += int(round(self._gaze[1]))
        ew, eh = 110, 100
        blink_squish = 1.0 - self._blink_t * 0.97
        pulse = 1.0 + m.eye_speech_pulse * self._open
        eh_now = max(3, int(eh * blink_squish * self._eye_squish * pulse))
        pts = [(cx, cy - eh_now // 2), (cx - ew // 2, cy + eh_now // 2), (cx + ew // 2, cy + eh_now // 2)]
        tilt = -self._eye_tilt if is_left else self._eye_tilt
        if abs(tilt) > 0.5:
            rad = math.radians(tilt)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            pts = [(int(cx + (x - cx) * cos_a - (y - cy) * sin_a),
                    int(cy + (x - cx) * sin_a + (y - cy) * cos_a)) for x, y in pts]
        self._draw_shape_glow(surf, pts, layers=6, spread=14)
        pygame.draw.polygon(surf, m.eye_color, pts)
        self._draw_drop_shadow(surf, pts[1])

    # ── Nose ─────────────────────────────────────────────
    def _draw_nose(self, surf) -> None:
        if self.assets.nose is not None:
            nc = self.manifest.nose
            ns = self.assets.nose
            ox, oy = self.assets.nose_offset
            surf.blit(ns, (nc.cx + ox - ns.get_width() // 2, nc.cy + oy - ns.get_height() // 2))
        elif self.manifest.draw_nose:
            m = self.manifest
            cx = m.canvas_w // 2
            cy = m.mouth.anchor_cy - 85
            pts = [(cx, cy - 22), (cx - 24, cy + 18), (cx + 24, cy + 18)]
            pygame.draw.polygon(surf, m.mouth.dark_color, pts)
            pygame.draw.polygon(surf, m.nose_color, pts, 2)

    # ── Procedural mouth ─────────────────────────────────
    def _draw_procedural_mouth(self, surf) -> None:
        mc = self.manifest.mouth
        cx, cy = mc.anchor_cx, mc.anchor_cy
        oa = self._open
        w = int(mc.min_w + (mc.max_w - mc.min_w) * self._width_t)
        open_h = int(90 * oa)

        if oa < 0.05:   # closed: thin bar
            pts = [(cx - w // 2, cy - 7), (cx + w // 2, cy - 7),
                   (cx + w // 2, cy + 7), (cx - w // 2, cy + 7)]
            self._draw_shape_glow(surf, pts, layers=4, spread=10)
            pygame.draw.polygon(surf, mc.color, pts)
            self._draw_drop_shadow(surf, (cx - w // 2, cy + 7))
            return

        if mc.style == "rounded" or self._rounded_blend >= 0.5:
            self._draw_oval(surf, mc, cx, cy, w, open_h)
        else:
            self._draw_toothed(surf, mc, cx, cy, w, open_h, oa)

    def _draw_toothed(self, surf, mc, cx, cy, w, open_h, oa) -> None:
        teeth_h = int(32 * oa)
        n = max(1, mc.n_teeth)
        top_pts, bot_pts = [], []
        for i in range(n * 2 + 1):
            x = cx - w // 2 + int(i * w / (n * 2))
            tooth = 0 if i % 2 == 0 else teeth_h
            top_pts.append((x, cy - open_h // 2 + tooth))
            bot_pts.append((x, cy + open_h // 2 - tooth))
        all_pts = top_pts + list(reversed(bot_pts))
        self._draw_shape_glow(surf, all_pts, layers=5, spread=12)
        pygame.draw.polygon(surf, mc.color, all_pts)
        self._draw_drop_shadow(surf, bot_pts[0])

    def _draw_oval(self, surf, mc, cx, cy, w, open_h) -> None:
        ow = max(20, int(w * 0.52))
        oh = max(10, open_h)
        pts = [(int(cx + ow // 2 * math.cos(a)), int(cy + oh // 2 * math.sin(a)))
               for a in (2 * math.pi * i / 24 for i in range(24))]
        self._draw_shape_glow(surf, pts, layers=5, spread=12)
        pygame.draw.ellipse(surf, mc.color, (cx - ow // 2, cy - oh // 2, ow, oh))
        self._draw_drop_shadow(surf, (cx - ow // 3, cy + oh // 3))
