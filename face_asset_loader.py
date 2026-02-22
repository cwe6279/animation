"""
face_asset_loader.py
====================
Loads artist-created face assets (PNG) from a face directory.

Directory structure:
  faces/
    <face_name>/
      face.json          <- manifest (required)
      face_base.png      <- full face background art (optional)
      eye_left.png       <- left eye art (optional)
      eye_right.png      <- right eye art (optional)
      mouth_sil.png      <- mouth state images (optional, any subset)
      mouth_pp.png
      mouth_aa.png
      mouth_oo.png
      mouth_ee.png
      mouth_ah.png

All art files are optional PNGs. Missing files are skipped gracefully.
If face_base is absent, only eyes + mouth render on black (for projection).
If a mouth state image is missing, falls back to nearest defined state.
If no mouth states are defined at all, renderer uses procedural mouth.
"""

from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import pygame
import numpy as np

from phoneme_scheduler import Viseme, Emotion


# ─────────────────────────────────────────────────────────
# FACE MANIFEST  (face.json schema)
# ─────────────────────────────────────────────────────────
@dataclass
class EyeConfig:
    image:      Optional[str] = None   # filename relative to face dir
    cx:         int  = 0
    cy:         int  = 0
    scale:      float = 1.0
    opacity:    float = 1.0            # 0.0=invisible, 1.0=fully opaque
    blink_axis: str  = "y"             # "y" = squish vertically (default)

@dataclass
class MouthConfig:
    anchor_cx: int   = 400
    anchor_cy: int   = 540
    max_w:     int   = 260
    min_w:     int   = 160
    scale:     float = 1.0             # scale mouth art PNGs around anchor point
    offset_x:  int   = 0              # pixel offset to shift art mouth PNGs
    offset_y:  int   = 0
    opacity:   float = 1.0            # 0.0=invisible, 1.0=fully opaque
    # Procedural overrides when no mouth images provided
    color:       Tuple = (255, 200, 0)
    dark_color:  Tuple = (10,  10, 10)
    style:       str   = "toothed"     # "toothed" | "rounded" | "simple"
    n_teeth:     int   = 5
    corner_radius: int = 0             # rounded corners on simple style

@dataclass
class NoseConfig:
    image:   Optional[str] = None   # filename relative to face dir (e.g. "nose.png")
    cx:      int   = 400
    cy:      int   = 440
    scale:   float = 1.0
    opacity: float = 1.0            # 0.0=invisible, 1.0=fully opaque

@dataclass
class FaceManifest:
    name:         str   = "custom"
    description:  str   = ""
    canvas_w:     int   = 800
    canvas_h:     int   = 800
    fps:          int   = 60
    bg_color:     Tuple = (10, 10, 20)

    # Base face art
    face_base:    Optional[str] = None  # PNG filename
    face_base_opacity: float = 1.0     # 0.0=invisible, 1.0=fully opaque
    face_color:   Tuple = (210, 100, 0)
    face_outline: Tuple = (140,  60, 0)
    glow_color:   Tuple = (255, 160, 0)
    glow_intensity: float = 1.0         # 0=no glow, 1=normal, 2=intense

    # Eyes
    eye_left:     EyeConfig = field(default_factory=EyeConfig)
    eye_right:    EyeConfig = field(default_factory=EyeConfig)
    eye_color:    Tuple = (255, 200, 0)  # used for procedural eyes
    blink:        bool  = True

    # Nose
    draw_nose:    bool  = False
    nose_color:   Tuple = (255, 200, 0)
    nose:         NoseConfig = field(default_factory=NoseConfig)

    # Mouth
    mouth:        MouthConfig = field(default_factory=MouthConfig)

    # Mouth state image filenames (keyed by viseme name)
    mouth_images: Dict[str, str] = field(default_factory=dict)

    # Procedural draw_stem (for pumpkin-type faces)
    draw_stem:    bool  = False
    stem_color:   Tuple = (60, 120, 20)

    @staticmethod
    def from_dict(d: dict) -> "FaceManifest":
        m = FaceManifest()
        m.name        = d.get("name", m.name)
        m.description = d.get("description", m.description)
        m.canvas_w    = d.get("canvas_w", m.canvas_w)
        m.canvas_h    = d.get("canvas_h", m.canvas_h)
        m.fps         = d.get("fps",      m.fps)
        m.bg_color    = tuple(d.get("bg_color",    m.bg_color))
        m.face_base   = d.get("face_base", None)
        m.face_base_opacity = float(d.get("face_base_opacity", m.face_base_opacity))
        m.face_color  = tuple(d.get("face_color",  m.face_color))
        m.face_outline= tuple(d.get("face_outline",m.face_outline))
        m.glow_color  = tuple(d.get("glow_color",  m.glow_color))
        m.glow_intensity = float(d.get("glow_intensity", m.glow_intensity))
        m.eye_color   = tuple(d.get("eye_color",   m.eye_color))
        m.blink       = d.get("blink", m.blink)
        m.draw_nose   = d.get("draw_nose", m.draw_nose)
        m.nose_color  = tuple(d.get("nose_color", m.nose_color))
        m.draw_stem   = d.get("draw_stem", m.draw_stem)
        m.stem_color  = tuple(d.get("stem_color",  m.stem_color))
        m.mouth_images= d.get("mouth_images", {})

        # Eye sub-configs
        for side in ("eye_left", "eye_right"):
            ed = d.get(side, {})
            ec = EyeConfig()
            ec.image  = ed.get("image",  None)
            ec.cx     = ed.get("cx",     280 if side == "eye_left" else 520)
            ec.cy     = ed.get("cy",     320)
            ec.scale   = float(ed.get("scale", 1.0))
            ec.opacity = float(ed.get("opacity", 1.0))
            setattr(m, side, ec)

        # Mouth sub-config
        md = d.get("mouth", {})
        mc = MouthConfig()
        mc.anchor_cx    = md.get("anchor_cx",    mc.anchor_cx)
        mc.anchor_cy    = md.get("anchor_cy",    mc.anchor_cy)
        mc.max_w        = md.get("max_w",        mc.max_w)
        mc.min_w        = md.get("min_w",        mc.min_w)
        mc.scale        = float(md.get("scale",  mc.scale))
        mc.offset_x     = md.get("offset_x",   mc.offset_x)
        mc.offset_y     = md.get("offset_y",   mc.offset_y)
        mc.opacity      = float(md.get("opacity", mc.opacity))
        mc.color        = tuple(md.get("color",       mc.color))
        mc.dark_color   = tuple(md.get("dark_color",  mc.dark_color))
        mc.style        = md.get("style",        mc.style)
        mc.n_teeth      = md.get("n_teeth",      mc.n_teeth)
        mc.corner_radius= md.get("corner_radius", mc.corner_radius)
        m.mouth         = mc

        # Nose sub-config
        nd = d.get("nose", {})
        nc = NoseConfig()
        nc.image   = nd.get("image",   None)
        nc.cx      = nd.get("cx",      nc.cx)
        nc.cy      = nd.get("cy",      nc.cy)
        nc.scale   = float(nd.get("scale",   nc.scale))
        nc.opacity = float(nd.get("opacity", nc.opacity))
        m.nose     = nc

        return m


# ─────────────────────────────────────────────────────────
# EMOTION PARAMETERS  (eye modulation per emotion)
# ─────────────────────────────────────────────────────────
# eye_squish: vertical scale (<1 = narrow/squint, >1 = wide)
# eye_tilt:   degrees rotation, positive = inner corners down (angry)
# blink_mult: multiplier on blink frequency
# speed:      lerp rate for transition (higher = snappier reaction)
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
    Viseme.AH:  ["ah",  "neutral","sil"],
}

@dataclass
class LoadedFaceAssets:
    manifest:    FaceManifest
    face_base:   Optional[pygame.Surface] = None
    eye_left:    Optional[pygame.Surface] = None
    eye_right:   Optional[pygame.Surface] = None
    nose:        Optional[pygame.Surface] = None
    mouth_surfs: Dict[str, pygame.Surface] = field(default_factory=dict)

    def get_mouth_surf(self, viseme: Viseme) -> Optional[pygame.Surface]:
        """Find best matching mouth surface for a viseme, with fallback."""
        for key in VISEME_TO_MOUTH_KEY.get(viseme, []):
            if key in self.mouth_surfs:
                return self.mouth_surfs[key]
        # Last resort: first available mouth image
        if self.mouth_surfs:
            return next(iter(self.mouth_surfs.values()))
        return None

    def has_mouth_art(self) -> bool:
        return bool(self.mouth_surfs)


def _load_image(path: str) -> Optional[pygame.Surface]:
    """Load a PNG image, return pygame Surface or None."""
    if not os.path.exists(path):
        return None
    try:
        surf = pygame.image.load(path)
        try:
            return surf.convert_alpha()
        except pygame.error:
            return surf  # display not yet initialized, convert later
    except Exception as e:
        print(f"[assets] PNG load error for {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────
# FACE ASSET LOADER
# ─────────────────────────────────────────────────────────
class FaceAssetLoader:
    """
    Loads a face from a directory containing face.json + art assets.
    Returns LoadedFaceAssets ready for the renderer.
    """

    def load(self, face_dir: str) -> LoadedFaceAssets:
        manifest_path = os.path.join(face_dir, "face.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No face.json found in {face_dir}")

        with open(manifest_path) as f:
            manifest = FaceManifest.from_dict(json.load(f))

        assets = LoadedFaceAssets(manifest=manifest)

        print(f"[assets] Loading face: {manifest.name}")

        # Face base art (optional — skipped if missing or not specified)
        if manifest.face_base:
            path = os.path.join(face_dir, manifest.face_base)
            assets.face_base = _load_image(path)
            if assets.face_base:
                print(f"[assets]   face_base: {manifest.face_base} OK")
            else:
                print(f"[assets]   face_base: {manifest.face_base} not found, skipping (eyes+mouth only)")

        # Eye art
        for side in ("eye_left", "eye_right"):
            ec = getattr(manifest, side)
            if ec.image:
                path = os.path.join(face_dir, ec.image)
                surf = _load_image(path)
                if surf:
                    # Scale if requested
                    if ec.scale != 1.0:
                        sw = int(surf.get_width()  * ec.scale)
                        sh = int(surf.get_height() * ec.scale)
                        surf = pygame.transform.scale(surf, (sw, sh))
                    setattr(assets, side, surf)
                    print(f"[assets]   {side}: {ec.image} OK")

        # Nose art (optional — placeholder for future use)
        nc = manifest.nose
        if nc.image:
            path = os.path.join(face_dir, nc.image)
            surf = _load_image(path)
            if surf:
                if nc.scale != 1.0:
                    sw = max(1, int(surf.get_width()  * nc.scale))
                    sh = max(1, int(surf.get_height() * nc.scale))
                    surf = pygame.transform.smoothscale(surf, (sw, sh))
                assets.nose = surf
                print(f"[assets]   nose: {nc.image} OK")

        # Mouth state images
        for vis_key, filename in manifest.mouth_images.items():
            path = os.path.join(face_dir, filename)
            surf = _load_image(path)
            if surf:
                assets.mouth_surfs[vis_key.lower()] = surf
                print(f"[assets]   mouth/{vis_key}: {filename} OK")

        print(f"[assets] Loaded {manifest.name} — "
              f"base={'art' if assets.face_base else 'none'}, "
              f"eyes={'art' if assets.eye_left else 'procedural'}, "
              f"nose={'art' if assets.nose else 'none'}, "
              f"mouth={'art' if assets.has_mouth_art() else 'procedural'}")

        return assets


# ─────────────────────────────────────────────────────────
# ASSET-AWARE FACE RENDERER
# ─────────────────────────────────────────────────────────
class AssetFaceRenderer:
    """
    Renders a face using loaded assets where available,
    falling back to procedural drawing for missing elements.
    Handles all 12 viseme states + blink + glow.
    """

    def __init__(self, assets: LoadedFaceAssets):
        self.assets   = assets
        self.manifest = assets.manifest
        m = self.manifest

        # Smooth animation state
        self._open           = 0.0
        self._width_t        = 0.8
        self._rounded_blend  = 0.0
        self._blink_t        = 0.0
        self._blink_next     = self._now() + np.random.uniform(2.0, 4.0)
        self.current_viseme  = Viseme.SIL
        self.current_viseme_label = "sil"

        # Cross-fade between mouth states (start with sil so mouth shows immediately)
        self._mouth_prev_surf: Optional[pygame.Surface] = None
        self._mouth_curr_surf: Optional[pygame.Surface] = assets.get_mouth_surf(Viseme.SIL)
        self._mouth_fade      = 1.0    # 0=prev, 1=curr

        # Emotion state (smoothed toward targets for natural transitions)
        self._emotion         = Emotion.NEUTRAL
        self._eye_squish      = 1.0   # vertical scale multiplier for eyes
        self._eye_tilt        = 0.0   # degrees rotation (+ = inner down)
        self._blink_mult      = 1.0   # blink frequency multiplier
        self.current_emotion_label = "neutral"

        # Pre-build canvas + glow surface
        self._canvas_w = m.canvas_w
        self._canvas_h = m.canvas_h
        self._gs = pygame.Surface((m.canvas_w, m.canvas_h), pygame.SRCALPHA)

    def _now(self): return __import__("time").time()

    # ── Public update ────────────────────────────────────
    def update(self, viseme: Viseme, dt: float, emotion: "Emotion" = None):
        from phoneme_scheduler import VISEME_PROPS
        props = VISEME_PROPS[viseme]
        self.current_viseme_label = props.label

        if viseme != self.current_viseme:
            # Kick off cross-fade to new mouth art (if using art mouths)
            new_surf = self.assets.get_mouth_surf(viseme)
            if new_surf and new_surf is not self._mouth_curr_surf:
                self._mouth_prev_surf = self._mouth_curr_surf
                self._mouth_curr_surf = new_surf
                self._mouth_fade      = 0.0
            self.current_viseme = viseme

        # Advance cross-fade (slower for smoother blending between mouth states)
        self._mouth_fade = min(1.0, self._mouth_fade + dt * 8.0)

        # Smooth procedural mouth params (softened rates for natural movement)
        spd_open  = 14.0 if props.open_amount > self._open   else 7.0
        spd_width = 11.0 if props.width_scale > self._width_t else 6.0
        self._open    += (props.open_amount - self._open)    * spd_open  * dt
        self._width_t += (props.width_scale - self._width_t) * spd_width * dt
        self._open    = max(0.0, min(1.0, self._open))
        self._width_t = max(0.0, min(1.0, self._width_t))

        target_round = 1.0 if props.rounded else 0.0
        self._rounded_blend += (target_round - self._rounded_blend) * 9.0 * dt

        # Emotion — smooth eye parameters toward targets (~250ms transition)
        if emotion is not None:
            self._emotion = emotion
            self.current_emotion_label = emotion.value
        ep = EMOTION_PARAMS.get(self._emotion, EMOTION_PARAMS[Emotion.NEUTRAL])
        emo_spd = ep["speed"]
        self._eye_squish += (ep["eye_squish"] - self._eye_squish) * emo_spd * dt
        self._eye_tilt   += (ep["eye_tilt"]   - self._eye_tilt)   * emo_spd * dt
        self._blink_mult += (ep["blink_mult"] - self._blink_mult) * emo_spd * dt

        # Blink (frequency modulated by emotion)
        now = self._now()
        if self.manifest.blink and now >= self._blink_next:
            self._blink_t    = 1.0
            interval = np.random.uniform(2.5, 5.5) / max(0.1, self._blink_mult)
            self._blink_next = now + interval
        if self._blink_t > 0:
            self._blink_t = max(0.0, self._blink_t - dt * 9.0)

    # ── Draw frame ───────────────────────────────────────
    def draw(self, surf: pygame.Surface):
        m = self.manifest
        surf.fill(m.bg_color)

        if self.assets.face_base:
            # Has body art: draw ambient glow + body + all features
            self._draw_ambient_glow(surf)
            if m.face_base_opacity < 1.0:
                fb = self.assets.face_base.copy()
                fb.set_alpha(int(m.face_base_opacity * 255))
                surf.blit(fb, (0, 0))
            else:
                surf.blit(self.assets.face_base, (0, 0))
            if m.draw_stem:
                self._draw_stem(surf)
            self._draw_eye_side(surf, m.eye_left,  self.assets.eye_left,  is_left=True)
            self._draw_eye_side(surf, m.eye_right, self.assets.eye_right, is_left=False)
            self._draw_nose_side(surf)
        else:
            # No body art: floating cut-outs on black with per-shape glow
            self._draw_eye_side(surf, m.eye_left,  self.assets.eye_left,  is_left=True)
            self._draw_eye_side(surf, m.eye_right, self.assets.eye_right, is_left=False)
            self._draw_nose_side(surf)

        # Mouth: art cross-fade or procedural
        if self.assets.has_mouth_art():
            self._draw_art_mouth(surf)
        else:
            self._draw_procedural_mouth(surf)

    # ── Per-shape glow halo (expands polygon outward) ────
    def _draw_shape_glow(self, surf, pts, color, layers=6, spread=14):
        gs = self._gs
        gs.fill((0, 0, 0, 0))
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        for i in range(layers, 0, -1):
            scale = 1.0 + (i / layers) * (spread / 60.0)
            alpha = int(80 * (i / layers) ** 1.4)
            expanded = [(int(cx + (x - cx) * scale), int(cy + (y - cy) * scale))
                        for x, y in pts]
            pygame.draw.polygon(gs, (*color, alpha), expanded)
        surf.blit(gs, (0, 0))

    # ── Drop shadow (lower-left offset glow) ─────────────
    def _draw_drop_shadow(self, surf, corner, color):
        gs = self._gs
        gs.fill((0, 0, 0, 0))
        for r in range(44, 0, -8):
            alpha = int(120 * (1 - r / 44.0) ** 1.5)
            pygame.draw.circle(gs, (*color, alpha), corner, r)
        surf.blit(gs, (0, 0))

    # ── Ambient glow (for faces with body art) ───────────
    def _draw_ambient_glow(self, surf):
        m  = self.manifest
        gs = self._gs
        gs.fill((0, 0, 0, 0))
        base_intensity = int(35 * m.glow_intensity + 40 * self._open * m.glow_intensity)
        cx = m.canvas_w  // 2
        cy = m.canvas_h  // 2
        rx = int(m.canvas_w * 0.42)
        ry = int(m.canvas_h * 0.46)
        for r in range(110, 0, -12):
            alpha = max(0, base_intensity - r // 3)
            pygame.draw.ellipse(gs, (*m.glow_color, alpha),
                (cx - rx - r, cy - ry - r, (rx + r)*2, (ry + r)*2))
        surf.blit(gs, (0, 0))

    # ── Stem ─────────────────────────────────────────────
    def _draw_stem(self, surf):
        m   = self.manifest
        cx  = m.canvas_w // 2
        top = m.canvas_h // 2 - int(m.canvas_h * 0.44)
        pts = [(cx-14, top+10),(cx-4, top-58),(cx+26, top-78),(cx+30, top-18),(cx+14, top+10)]
        pygame.draw.polygon(surf, m.stem_color, pts)
        pygame.draw.polygon(surf, (40, 90, 10), pts, 2)

    # ── Eye (art or procedural triangle) ─────────────────
    def _draw_eye_side(self, surf, ec: "EyeConfig", eye_surf: Optional[pygame.Surface],
                       is_left: bool = True):
        if eye_surf:
            self._draw_art_eye(surf, ec, eye_surf, is_left)
        else:
            self._draw_procedural_eye(surf, ec.cx, ec.cy, is_left)

    def _draw_art_eye(self, surf, ec: "EyeConfig", eye_surf: pygame.Surface,
                      is_left: bool):
        """Blit art eye with blink squish, emotion squish/tilt, and opacity."""
        ow = eye_surf.get_width()
        oh = eye_surf.get_height()
        # Blink squish + emotion squish
        blink_squish = max(0.03, 1.0 - self._blink_t * 0.97)
        emo_squish   = self._eye_squish
        total_squish = blink_squish * emo_squish
        new_h = max(2, int(oh * total_squish))
        scaled = pygame.transform.scale(eye_surf, (ow, new_h))
        # Emotion tilt: left eye -tilt, right eye +tilt
        # Positive eye_tilt = angry (inner corners down, V-shape)
        tilt = -self._eye_tilt if is_left else self._eye_tilt
        if abs(tilt) > 0.5:
            scaled = pygame.transform.rotate(scaled, tilt)
        if ec.opacity < 1.0:
            scaled.set_alpha(int(ec.opacity * 255))
        # Center on ec.cx/cy (rotate may change surface size)
        blit_x = ec.cx - scaled.get_width()  // 2
        blit_y = ec.cy - scaled.get_height() // 2
        surf.blit(scaled, (blit_x, blit_y))

    def _draw_procedural_eye(self, surf, cx, cy, is_left: bool):
        m  = self.manifest
        ew, eh = 110, 100
        # Blink squish + emotion squish
        blink_squish = 1.0 - self._blink_t * 0.97
        emo_squish   = self._eye_squish
        eh_now = max(3, int(eh * blink_squish * emo_squish))
        tip = (cx,           cy - eh_now // 2)
        bl  = (cx - ew // 2, cy + eh_now // 2)
        br  = (cx + ew // 2, cy + eh_now // 2)
        pts = [tip, bl, br]
        # Emotion tilt: rotate points around eye center
        # Positive eye_tilt = angry (inner corners down, V-shape)
        tilt = -self._eye_tilt if is_left else self._eye_tilt
        if abs(tilt) > 0.5:
            rad = math.radians(tilt)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            pts = [(int(cx + (x - cx) * cos_a - (y - cy) * sin_a),
                    int(cy + (x - cx) * sin_a + (y - cy) * cos_a))
                   for x, y in pts]
        self._draw_shape_glow(surf, pts, m.glow_color, layers=6, spread=14)
        pygame.draw.polygon(surf, m.eye_color, pts)
        # Drop shadow on lower-left corner
        self._draw_drop_shadow(surf, pts[1], m.glow_color)

    # ── Nose (art or procedural) ──────────────────────────
    def _draw_nose_side(self, surf):
        """Render nose: art PNG if available, procedural if draw_nose=true, else skip."""
        if self.assets.nose:
            self._draw_art_nose(surf)
        elif self.manifest.draw_nose:
            self._draw_procedural_nose(surf)

    def _draw_art_nose(self, surf):
        nc = self.manifest.nose
        ns = self.assets.nose
        blit_x = nc.cx - ns.get_width() // 2
        blit_y = nc.cy - ns.get_height() // 2
        if nc.opacity < 1.0:
            ns = ns.copy()
            ns.set_alpha(int(nc.opacity * 255))
        surf.blit(ns, (blit_x, blit_y))

    def _draw_procedural_nose(self, surf):
        m  = self.manifest
        cx = m.canvas_w // 2
        cy = m.mouth.anchor_cy - 85
        pts = [(cx, cy-22), (cx-24, cy+18), (cx+24, cy+18)]
        pygame.draw.polygon(surf, m.mouth.dark_color, pts)
        pygame.draw.polygon(surf, m.nose_color, pts, 2)

    # ── Art mouth: instant swap (no cross-fade — overlapping PNGs look unnatural) ──
    def _draw_art_mouth(self, surf):
        mc = self.manifest.mouth
        if self._mouth_curr_surf:
            curr = self._scale_mouth_surf(self._mouth_curr_surf, mc)
            if mc.opacity < 1.0:
                curr.set_alpha(int(mc.opacity * 255))
            surf.blit(curr, self._mouth_blit_pos(curr, mc))

    def _scale_mouth_surf(self, src, mc):
        """Scale a mouth surface around the anchor point."""
        if mc.scale == 1.0:
            return src.copy()
        sw = max(1, int(src.get_width()  * mc.scale))
        sh = max(1, int(src.get_height() * mc.scale))
        return pygame.transform.smoothscale(src, (sw, sh))

    def _mouth_blit_pos(self, scaled, mc):
        """Blit position: scale pivot + offset shift."""
        bx, by = mc.offset_x, mc.offset_y
        if mc.scale != 1.0:
            ox, oy = mc.anchor_cx, mc.anchor_cy
            sx, sy = ox * mc.scale, oy * mc.scale
            bx += int(ox - sx)
            by += int(oy - sy)
        return (bx, by)

    # ── Procedural mouth ─────────────────────────────────
    def _draw_procedural_mouth(self, surf):
        m   = self.manifest
        mc  = m.mouth
        cx, cy = mc.anchor_cx, mc.anchor_cy
        oa     = self._open
        rb     = self._rounded_blend
        w      = int(mc.min_w + (mc.max_w - mc.min_w) * self._width_t)
        open_h = int(90 * oa)

        if oa < 0.05:
            # Closed mouth — thin line with subtle glow
            pts = [(cx - w//2, cy - 7), (cx + w//2, cy - 7),
                   (cx + w//2, cy + 7), (cx - w//2, cy + 7)]
            self._draw_shape_glow(surf, pts, m.glow_color, layers=4, spread=10)
            pygame.draw.polygon(surf, mc.color, pts)
            self._draw_drop_shadow(surf, (cx - w//2, cy + 7), m.glow_color)
            return

        style = mc.style
        if style == "toothed" and rb < 0.5:
            self._draw_toothed(surf, mc, cx, cy, w, open_h, oa)
        elif style == "rounded" or rb >= 0.5:
            self._draw_oval(surf, mc, cx, cy, w, open_h)
        else:
            self._draw_toothed(surf, mc, cx, cy, w, open_h, oa)

    def _draw_toothed(self, surf, mc, cx, cy, w, open_h, oa):
        m = self.manifest
        teeth_h = int(32 * oa)
        n = mc.n_teeth
        top_pts, bot_pts = [], []
        for i in range(n*2+1):
            x = cx - w//2 + int(i*w/(n*2))
            top_pts.append((x, cy - open_h//2 + (0 if i%2==0 else teeth_h)))
            bot_pts.append((x, cy + open_h//2 - (0 if i%2==0 else teeth_h)))
        all_pts = top_pts + list(reversed(bot_pts))
        self._draw_shape_glow(surf, all_pts, m.glow_color, layers=5, spread=12)
        pygame.draw.polygon(surf, mc.color, all_pts)
        # Drop shadow on lower-left corner
        self._draw_drop_shadow(surf, bot_pts[0], m.glow_color)

    def _draw_oval(self, surf, mc, cx, cy, w, open_h):
        m    = self.manifest
        ow   = max(20, int(w * 0.52))
        oh   = max(10, open_h)
        # Build polygon for glow
        pts = []
        for i in range(24):
            angle = 2 * math.pi * i / 24
            pts.append((int(cx + ow // 2 * math.cos(angle)),
                        int(cy + oh // 2 * math.sin(angle))))
        self._draw_shape_glow(surf, pts, m.glow_color, layers=5, spread=12)
        pygame.draw.ellipse(surf, mc.color, (cx - ow//2, cy - oh//2, ow, oh))
        # Drop shadow on lower-left
        self._draw_drop_shadow(surf, (cx - ow//3, cy + oh//3), m.glow_color)
