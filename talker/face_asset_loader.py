"""
talker/face_asset_loader.py
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
from pygame import gfxdraw

from .phoneme_scheduler import Emotion, VISEME_PROPS, Viseme


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
    style:      str = "toothed"     # "toothed" | "rounded" | "grin" (carved smile, corners up, goofy teeth)
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
    # "halo": soft light spills outward around each cut-out (classic projection look).
    # "inner": crisp cut edges lit from inside, a hot core fading to the edge colour,
    #          like a candle behind a carved pumpkin. core_color defaults to the shape
    #          colour pushed toward white; rim_color draws a thin cut-edge line;
    #          light_offset moves the hot spot down (0.15 = a little below centre).
    glow_style:   str = "halo"
    core_color:   Optional[Tuple] = None
    rim_color:    Optional[Tuple] = None
    light_offset: float = 0.15
    # inner style only: the cut has depth. The lit shape is inset by cut_depth px and the
    # inner wall of the shell shows along the other side in wall_color (pale, lit yellow),
    # which reads as 3D. [0, 0] turns it off.
    cut_depth:    Tuple = (8, 6)
    wall_color:   Tuple = (255, 238, 170)

    eye_left:  EyeConfig = field(default_factory=EyeConfig)
    eye_right: EyeConfig = field(default_factory=EyeConfig)
    eye_color: Tuple = (255, 200, 0)    # procedural eyes
    draw_eyes: bool = True              # false = no eyes at all (a voice-only character)
    blink:     bool = True
    eye_speech_pulse: float = 0.0       # eyes grow by this fraction when the mouth is fully open
    eye_lids: bool = False              # image/procedural eyes: emotion as lid cuts (crescents, slants) and lid blinks
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

    # Live textured eyes (see talker/textured_eye.py). When set, both eyes are composed
    # per frame from the parts in `dir` (relative to the face folder) instead of
    # eye_left/eye_right images; the left eye is the mirror of the right.
    #   {"dir": "eye", "size": 224, "lid_open": 0.55, "gaze_radius": 0.35,
    #    "pupil": [0.12, 0.22, 0.40], "lid_tracking": 0.35}
    textured_eye: Dict = field(default_factory=dict)

    # Defaults for the voice loop (CLI flags override). voices is keyed by TTS
    # backend name: {"elevenlabs": "<voice id>", "edge": "en-US-AriaNeural"};
    # tts_model picks the ElevenLabs model (e.g. "eleven_v3").
    voices:     Dict[str, str] = field(default_factory=dict)
    tts_model:  Optional[str] = None
    voice_speed: Optional[float] = None # speaking rate multiplier (Flash / edge; v3 ignores)
    character:  str = ""                # personality; loaded from character.md next to face.json
    wake_words: List[str] = field(default_factory=list)   # wake mode: names that start a conversation
    sounds: str = "sounds"              # folder of sound effects next to face.json ({{sfx name}})
    body: Dict = field(default_factory=dict)   # {"moves": ["nod", ...]}: what {{move name}} may ask for
    sleep_words: List[str] = field(default_factory=list)  # wake mode: short phrases that end it at once

    _KNOWN_TOP = {
        "name", "description", "canvas_w", "canvas_h", "fps", "bg_color", "face_base",
        "face_base_opacity", "face_color", "face_outline", "glow_color", "glow_intensity",
        "glow_style", "core_color", "rim_color", "light_offset", "cut_depth", "wall_color", "eye_left", "eye_right", "eye_color", "draw_eyes", "blink", "blink_interval", "blink_speed",
        "eye_speech_pulse", "eye_lids", "gaze", "draw_nose", "nose_color", "nose",
        "mouth", "mouth_images", "draw_stem", "stem_color", "voices", "tts_model", "voice_speed", "character", "textured_eye", "wake_words", "sleep_words", "sounds", "body",
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
        m.glow_style  = str(d.get("glow_style", m.glow_style)).lower()
        if m.glow_style not in ("halo", "inner"):
            print(f"[assets] warning: glow_style {m.glow_style!r} unknown, using 'halo'")
            m.glow_style = "halo"
        m.core_color  = tuple(d["core_color"]) if d.get("core_color") else None
        m.rim_color   = tuple(d["rim_color"]) if d.get("rim_color") else None
        m.light_offset = float(d.get("light_offset", m.light_offset))
        m.cut_depth   = tuple(int(x) for x in d.get("cut_depth", m.cut_depth))
        m.wall_color  = tuple(d.get("wall_color", m.wall_color))
        m.eye_color   = tuple(d.get("eye_color", m.eye_color))
        m.blink       = bool(d.get("blink", m.blink))
        m.draw_eyes   = bool(d.get("draw_eyes", m.draw_eyes))
        m.eye_speech_pulse = float(d.get("eye_speech_pulse", m.eye_speech_pulse))
        m.eye_lids = bool(d.get("eye_lids", m.eye_lids))
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
        m.voice_speed = float(d["voice_speed"]) if d.get("voice_speed") else None
        m.textured_eye = dict(d.get("textured_eye") or {})
        m.wake_words = [str(w) for w in (d.get("wake_words") or [])]
        m.sleep_words = [str(w) for w in (d.get("sleep_words") or [])]
        m.sounds = str(d.get("sounds", m.sounds))
        m.body = dict(d.get("body") or {})
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


# Lid cuts per emotion for eye_lids faces, as fractions of the eye's solid core:
#   upper cover, lower cover, tilt, lower-lid arch, upper-lid corner drop.
# tilt > 0 covers the inner corner more (angry V); tilt < 0 droops the outer corner (sad).
# arch > 0 bows the lower lid upward in the middle; corner drop lowers the upper lid at
# both corners. Together they make an even, upward-bowed crescent (happy).
EMOTION_LIDS = {
    Emotion.NEUTRAL:  (0.00, 0.00,  0.00, 0.00, 0.00),
    Emotion.HAPPY:    (0.00, 0.18,  0.00, 0.30, 0.30),
    Emotion.ANGRY:    (0.40, 0.05,  0.30, 0.00, 0.00),
    Emotion.ANNOYED:  (0.30, 0.05,  0.10, 0.00, 0.00),
    Emotion.SAD:      (0.28, 0.05, -0.30, 0.00, 0.00),
    Emotion.SURPRISE: (0.00, 0.00,  0.00, 0.00, 0.00),
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
    # Live textured eyes (textured_eye.TexturedEye), right and left, or None
    textured: Optional[Tuple[object, object]] = None
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
        # Personality lives in character.md next to the art when present; the
        # shared system prompt is concatenated with it at runtime.
        char_path = os.path.join(face_dir, "character.md")
        if os.path.isfile(char_path):
            with open(char_path, encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                manifest.character = text
                print(f"[assets]   character.md: {len(text.split())} words")
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

        te = manifest.textured_eye
        if te.get("dir"):
            from .textured_eye import TexturedEyeAssets, TexturedEye
            folder = path_of(te["dir"])
            size = int(te.get("size", 224))
            lid_open = float(te.get("lid_open", 0.55))
            rim = float(te.get("rim", 1.0))
            try:
                right = TexturedEye(TexturedEyeAssets(folder, size, mirror=False), lid_open=lid_open, rim=rim)
                left = TexturedEye(TexturedEyeAssets(folder, size, mirror=True), lid_open=lid_open, rim=rim)
                assets.textured = (right, left)
                print(f"[assets]   textured eyes from {te['dir']}/ at {size}px")
            except Exception as e:
                print(f"[assets]   textured eyes failed ({e}); falling back to image/procedural eyes")

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
              f"eyes={'textured' if assets.textured else ('art' if assets.eye_left else 'procedural')}, "
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


def _lit_polygon(pts: List[Tuple[int, int]], edge_color, core_color, rim_color=None,
                 light_offset: float = 0.15, falloff: float = 1.6,
                 cut_depth: Tuple[int, int] = (0, 0), wall_color=None
                 ) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """A cut-out lit from inside: a radial gradient from a hot core to the edge
    colour, clipped to the polygon with an anti-aliased (crisp) edge. Returns the
    sprite and its blit origin. Small surfaces, so cheap enough per frame."""
    x0 = int(min(p[0] for p in pts)) - 1
    y0 = int(min(p[1] for p in pts)) - 1
    w = int(max(p[0] for p in pts)) - x0 + 2
    h = int(max(p[1] for p in pts)) - y0 + 2
    w, h = max(2, w), max(2, h)
    lx = sum(p[0] for p in pts) / len(pts)
    ly = sum(p[1] for p in pts) / len(pts) + h * light_offset
    rmax = max(1.0, max(math.hypot(x - lx, y - ly) for x, y in pts))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx + x0 - lx, yy + y0 - ly) / rmax
    t = np.clip(1.0 - d, 0.0, 1.0) ** falloff                    # 1 at the core
    e = np.array(edge_color[:3], dtype=np.float32)
    c = np.array(core_color[:3], dtype=np.float32)
    rgb = (e + (c - e) * t[..., None]).astype(np.uint8)          # (h, w, 3)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    px = pygame.surfarray.pixels3d(surf)
    px[...] = rgb.transpose(1, 0, 2)
    del px
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[...] = 255
    del pa
    local = [(x - x0, y - y0) for x, y in pts]
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    gfxdraw.filled_polygon(mask, local, (255, 255, 255, 255))
    gfxdraw.aapolygon(mask, local, (255, 255, 255, 255))
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    dx, dy = cut_depth
    if (dx or dy) and wall_color is not None:
        # inner wall: the part of the opening not covered by the shape shifted by the
        # cut depth. Flat pale colour, like light on the thickness of the shell.
        wall = mask.copy()
        inset = pygame.Surface((w, h), pygame.SRCALPHA)
        shifted = [(x + dx, y + dy) for x, y in local]
        gfxdraw.filled_polygon(inset, shifted, (255, 255, 255, 255))
        gfxdraw.aapolygon(inset, shifted, (255, 255, 255, 255))
        wall.blit(inset, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        wall.fill((*wall_color[:3], 255), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(wall, (0, 0))
    if rim_color is not None:
        pygame.draw.polygon(surf, rim_color[:3], local, 1)
        gfxdraw.aapolygon(surf, local, rim_color[:3])
    return surf, (x0, y0)


def _toward_white(color, amount: float = 0.8) -> Tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in color[:3])


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
        self._k = assets.manifest.canvas_w / 800.0   # procedural shapes scale with the canvas
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

        # Lid state for eye_lids faces (smoothed toward EMOTION_LIDS targets)
        self._lids = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._lid_mask_cache: "collections.OrderedDict[tuple, pygame.Surface]" = collections.OrderedDict()
        self._core_span_cache: dict = {}

        # Live textured eyes: shared motion model (both eyes move together)
        self._eye_motion = None
        if assets.textured is not None:
            from .textured_eye import EyeMotion, EyeMotionConfig
            te = m.textured_eye
            cfg = EyeMotionConfig()
            if "gaze_radius" in te:
                cfg.gaze_radius = float(te["gaze_radius"])
            if "lid_tracking" in te:
                cfg.lid_tracking = float(te["lid_tracking"])
            if "pupil" in te:
                pmin, pbase, pmax = te["pupil"]
                cfg.pupil_min, cfg.pupil_base, cfg.pupil_max = float(pmin), float(pbase), float(pmax)
            self._eye_motion = EyeMotion(cfg)

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
        if self.manifest.eye_lids:
            # lids carry the expression; keep only a mild squish/tilt underneath
            squish_t = 1.0 + (ep["eye_squish"] - 1.0) * 0.4
            tilt_t = ep["eye_tilt"] * 0.3
            for i, target in enumerate(EMOTION_LIDS.get(self._emotion, EMOTION_LIDS[Emotion.NEUTRAL])):
                self._lids[i] += (target - self._lids[i]) * k
        else:
            squish_t, tilt_t = ep["eye_squish"], ep["eye_tilt"]
        self._eye_squish += (squish_t - self._eye_squish) * k
        self._eye_tilt += (tilt_t - self._eye_tilt) * k
        self._blink_mult += (ep["blink_mult"] - self._blink_mult) * k

        if self._eye_motion is not None:
            # textured eyes own their blink and glances
            self._eye_motion.update(dt, self._emotion, blink_scale=max(0.1, self._blink_mult))
            if viseme is not Viseme.SIL:
                self._last_speech = time.monotonic()
            return

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
        if not m.draw_eyes:
            pass                                   # a voice-only character
        elif self.assets.textured is not None:
            self._draw_textured_eye(surf, m.eye_left, self.assets.textured[1], is_left=True)
            self._draw_textured_eye(surf, m.eye_right, self.assets.textured[0], is_left=False)
        else:
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

    def _draw_lit(self, surf, pts, color) -> None:
        """glow_style 'inner': crisp cut-out lit from within (no halo, no shadow)."""
        m = self.manifest
        core = m.core_color or _toward_white(color, min(1.0, 0.55 + 0.25 * m.glow_intensity))
        depth = (int(round(m.cut_depth[0] * self._k)), int(round(m.cut_depth[1] * self._k)))
        sprite, pos = _lit_polygon(pts, color, core, m.rim_color, m.light_offset,
                                   cut_depth=depth, wall_color=m.wall_color)
        surf.blit(sprite, pos)

    def _draw_shape(self, surf, pts, color, layers=5, spread=12, shadow_corner=None) -> None:
        """A procedural cut-out in the face's glow style."""
        if self.manifest.glow_style == "inner":
            self._draw_lit(surf, pts, color)
            return
        self._draw_shape_glow(surf, pts, layers=layers, spread=int(spread * self._k))
        pygame.draw.polygon(surf, color, pts)
        if shadow_corner is not None:
            self._draw_drop_shadow(surf, shadow_corner)

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

    def _core_span(self, surf: pygame.Surface) -> Tuple[float, float]:
        """
        Vertical extent (as fractions of image height) of the eye's solid core,
        i.e. rows with strongly opaque pixels, ignoring soft glow. Lids are cut
        relative to this span so a halo around the eye doesn't skew them.
        """
        key = id(surf)
        cached = self._core_span_cache.get(key)
        if cached is not None:
            return cached
        alpha = pygame.surfarray.pixels_alpha(surf)
        rows = np.where((alpha >= 200).any(axis=0))[0]
        del alpha
        h = surf.get_height()
        span = (rows[0] / h, (rows[-1] + 1) / h) if rows.size else (0.0, 1.0)
        self._core_span_cache[key] = span
        return span

    def _lid_mask(self, w: int, h: int, upper: float, lower: float, tilt: float, arch: float,
                  is_left: bool, span: Tuple[float, float] = (0.0, 1.0),
                  corner_drop: float = 0.0) -> Optional[pygame.Surface]:
        """
        Multiply-mask that hides the parts of an eye image covered by the lids.
        The upper lid is a straight cut whose inner end sits lower by `tilt`;
        the lower lid rises by `lower` and bows upward by `arch` in the middle.
        Cached by quantized parameters.
        """
        q = (w, h, round(upper, 2), round(lower, 2), round(tilt, 2), round(arch, 2), is_left,
             round(span[0], 3), round(span[1], 3), round(corner_drop, 2))
        m = self._lid_mask_cache.get(q)
        if m is not None:
            self._lid_mask_cache.move_to_end(q)
            return m
        if upper < 0.01 and lower < 0.01 and corner_drop < 0.01:
            return None
        m = pygame.Surface((w, h), pygame.SRCALPHA)
        m.fill((255, 255, 255, 255))
        top, bottom = span[0] * h, span[1] * h
        core = max(1.0, bottom - top)
        # x runs from the outer corner (0) to the inner corner (1) for the right eye;
        # mirror for the left so "inner" always means toward the nose.
        n = 24
        xs = [i / n for i in range(n + 1)]
        if upper > 0.01 or corner_drop > 0.01:
            pts = [(0, 0), (w, 0)]
            for x in reversed(xs):
                inner = x if not is_left else 1.0 - x
                edge = (2 * x - 1) ** 2                          # 0 in the middle, 1 at the corners
                cover = upper + tilt * (inner - 0.5) + corner_drop * edge
                pts.append((int(x * w), int(top + max(0.0, cover) * core)))
            pygame.draw.polygon(m, (0, 0, 0, 0), pts)
        if lower > 0.01:
            pts = [(0, h), (w, h)]
            for x in reversed(xs):
                bow = arch * (1.0 - (2 * x - 1) ** 2)          # 0 at corners, arch in the middle
                cover = lower + bow
                pts.append((int(x * w), int(bottom - max(0.0, cover) * core)))
            pygame.draw.polygon(m, (0, 0, 0, 0), pts)
        self._lid_mask_cache[q] = m
        if len(self._lid_mask_cache) > 64:
            self._lid_mask_cache.popitem(last=False)
        return m

    def _draw_art_eye(self, surf, ec: EyeConfig, eye_surf: pygame.Surface,
                      offset: Tuple[int, int], is_left: bool) -> None:
        lids = self.manifest.eye_lids
        blink_squish = 1.0 if lids else max(0.03, 1.0 - self._blink_t * 0.97)
        pulse = 1.0 + self.manifest.eye_speech_pulse * self._open
        new_h = max(2, int(eye_surf.get_height() * blink_squish * self._eye_squish * pulse))
        tilt = -self._eye_tilt if is_left else self._eye_tilt   # + = inner corners down
        img = self._eye_variant(eye_surf, new_h, tilt, ec.opacity)
        if lids:
            u, lo, tl, arch, drop = self._lids
            # blink: both lids meet in the middle
            u = u + (1.0 - u) * self._blink_t * 0.62
            lo = lo + (1.0 - lo) * self._blink_t * 0.45
            mask = self._lid_mask(img.get_width(), img.get_height(), u, lo, tl, arch, is_left,
                                  self._core_span(img), corner_drop=drop * (1.0 - self._blink_t))
            if mask is not None:
                img = img.copy()
                img.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        cx = ec.cx + offset[0] + int(round(self._gaze[0]))
        cy = ec.cy + offset[1] + int(round(self._gaze[1]))
        surf.blit(img, (cx - img.get_width() // 2, cy - img.get_height() // 2))

    def _draw_textured_eye(self, surf, ec: EyeConfig, eye, is_left: bool) -> None:
        """Compose the eye from its parts, then apply our emotion squish/tilt and scale."""
        em = self._eye_motion
        gaze = (float(em.gaze[0]), float(em.gaze[1]))
        img = eye.surface(gaze, em.pupil, em.blink, em.upper_lid_extra)
        pulse = 1.0 + self.manifest.eye_speech_pulse * self._open
        scale = ec.scale * pulse
        w = max(2, int(img.get_width() * scale))
        h = max(2, int(img.get_height() * scale * self._eye_squish))
        if (w, h) != img.get_size():
            img = pygame.transform.smoothscale(img, (w, h)) if scale < 1.0 else pygame.transform.scale(img, (w, h))
        tilt = -self._eye_tilt if is_left else self._eye_tilt
        if abs(tilt) > 0.5:
            img = pygame.transform.rotate(img, tilt)
        if ec.opacity < 1.0:
            img.set_alpha(int(ec.opacity * 255))
        surf.blit(img, (ec.cx - img.get_width() // 2, ec.cy - img.get_height() // 2))

    def _draw_procedural_eye(self, surf, cx, cy, is_left: bool) -> None:
        m = self.manifest
        cx += int(round(self._gaze[0]))
        cy += int(round(self._gaze[1]))
        ew, eh = int(110 * self._k), int(100 * self._k)
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
        self._draw_shape(surf, pts, m.eye_color, layers=6, spread=14, shadow_corner=pts[1])

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
        k = self._k
        w = int(mc.min_w + (mc.max_w - mc.min_w) * self._width_t)
        open_h = int(90 * k * oa)

        if mc.style == "rounded" or self._rounded_blend >= 0.5:
            if oa >= 0.05:
                self._draw_oval(surf, mc, cx, cy, w, open_h)
                return
        elif mc.style == "grin":            # has its own closed shape
            self._draw_grin(surf, mc, cx, cy, w, open_h)
            return
        elif oa >= 0.05:
            self._draw_toothed(surf, mc, cx, cy, w, open_h, oa)
            return

        t = int(7 * k)                      # closed: thin bar
        pts = [(cx - w // 2, cy - t), (cx + w // 2, cy - t),
               (cx + w // 2, cy + t), (cx - w // 2, cy + t)]
        self._draw_shape(surf, pts, mc.color, layers=4, spread=10, shadow_corner=(cx - w // 2, cy + t))

    def _draw_grin(self, surf, mc, cx, cy, w, open_h) -> None:
        """A carved smile: corners turned up, a curved band at rest that opens from the
        middle while speaking. The teeth are part of the cut line: the top edge steps
        down and back up around each tooth left uncut, the bottom edge steps up, so the
        light, the inner wall and the rim all follow one outline, as on a real pumpkin."""
        k = self._k
        lift = int(w * 0.28)                          # how far the corners rise (a wide carved grin)
        top_c = cy - int(6 * k) - int(open_h * 0.45)  # centre of the top edge
        bot_c = cy + int(w * 0.19) + int(open_h * 0.6)  # centre of the bottom edge: a fat crescent at rest
        base = cy - lift

        def edge_y(x: float, centre: int) -> int:     # quadratic from corner to centre
            u = (x - cx) / (w / 2)
            return int(base + (centre - base) * (1.0 - u * u))

        # teeth: (edge, position across the mouth -0.5..0.5, width as a fraction of w)
        layout = [("top", -0.22, 0.14), ("bottom", 0.11, 0.10), ("top", 0.06, 0.07),
                  ("bottom", -0.13, 0.08)][:max(0, mc.n_teeth)]

        def edge_path(edge: str, centre: int, other: int) -> List[Tuple[int, int]]:
            """Left to right along one edge, detouring around each tooth on it."""
            sign = 1 if edge == "top" else -1         # teeth hang down from the top, rise from the bottom
            spans = sorted((cx + (fx - fw / 2) * w, cx + (fx + fw / 2) * w)
                           for e, fx, fw in layout if e == edge)
            pts: List[Tuple[int, int]] = []
            x = cx - w / 2
            step = w / 40.0
            for x0, x1 in spans:
                while x < x0:
                    pts.append((int(x), edge_y(x, centre)))
                    x += step
                mid = (x0 + x1) / 2
                band = abs(edge_y(mid, other) - edge_y(mid, centre))
                th = int(min(band * 0.55, (26 + open_h * 0.35) * k))
                y0, y1 = edge_y(x0, centre), edge_y(x1, centre)
                pts += [(int(x0), y0), (int(x0), y0 + sign * th), (int(x1), y1 + sign * th), (int(x1), y1)]
                x = x1 + step / 2
            while x <= cx + w / 2:
                pts.append((int(x), edge_y(x, centre)))
                x += step
            pts.append((cx + w // 2, base))
            return pts

        top = edge_path("top", top_c, bot_c)
        bot = edge_path("bottom", bot_c, top_c)
        pts = top + list(reversed(bot))
        self._draw_shape(surf, pts, mc.color, layers=5, spread=12, shadow_corner=bot[2])

    def _draw_toothed(self, surf, mc, cx, cy, w, open_h, oa) -> None:
        teeth_h = int(32 * self._k * oa)
        n = max(1, mc.n_teeth)
        top_pts, bot_pts = [], []
        for i in range(n * 2 + 1):
            x = cx - w // 2 + int(i * w / (n * 2))
            tooth = 0 if i % 2 == 0 else teeth_h
            top_pts.append((x, cy - open_h // 2 + tooth))
            bot_pts.append((x, cy + open_h // 2 - tooth))
        all_pts = top_pts + list(reversed(bot_pts))
        self._draw_shape(surf, all_pts, mc.color, layers=5, spread=12, shadow_corner=bot_pts[0])

    def _draw_oval(self, surf, mc, cx, cy, w, open_h) -> None:
        ow = max(20, int(w * 0.52))
        oh = max(10, open_h)
        pts = [(int(cx + ow // 2 * math.cos(a)), int(cy + oh // 2 * math.sin(a)))
               for a in (2 * math.pi * i / 24 for i in range(24))]
        if self.manifest.glow_style == "inner":
            self._draw_lit(surf, pts, mc.color)
            return
        self._draw_shape_glow(surf, pts, layers=5, spread=int(12 * self._k))
        pygame.draw.ellipse(surf, mc.color, (cx - ow // 2, cy - oh // 2, ow, oh))
        self._draw_drop_shadow(surf, (cx - ow // 3, cy + oh // 3))
