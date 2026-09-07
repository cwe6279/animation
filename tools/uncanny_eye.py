"""
uncanny_eye.py — render a Talker eye image from an Adafruit "Uncanny Eyes" design.

    python tools/uncanny_eye.py <uncanny convert/<eye> folder> <out_dir> [--size 400] [--pupil 0.22]

Reads the design's source parts (iris.png polar strip, pupilMap.png distance
field, lid-upper.png / lid-lower.png masks, sclera.png) and composes one open
eye, the way the microcontroller sketch does per pixel: angle and distance
look up the iris strip, the pupil is where distance is under a threshold, and
the lids trim the outline. Writes eye_left.png and eye_right.png (mirrored)
at the requested size with a transparent background.

Uncanny Eyes: https://github.com/adafruit/uncanny_eyes (MIT, Adafruit / Phillip Burgess).
"""

from __future__ import annotations

import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageFilter


def _load_gray(path: str, size: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def render_eye(src: str, size: int = 400, pupil: float = 0.22, lid_open: float = 0.55,
               highlight: bool = True) -> Image.Image:
    iris = np.asarray(Image.open(os.path.join(src, "iris.png")).convert("RGB"), dtype=np.float32) / 255.0
    ih, iw, _ = iris.shape                     # iris strip: x = angle, y = distance from pupil edge outward
    has_sclera = os.path.exists(os.path.join(src, "sclera.png"))
    sclera = None
    if has_sclera:
        sc = Image.open(os.path.join(src, "sclera.png")).convert("RGB").resize((size, size), Image.BILINEAR)
        sclera = np.asarray(sc, dtype=np.float32) / 255.0
        if sclera.max() < 0.05:
            sclera = None                      # all-black sclera = the iris fills the eye

    # Distance field: pupilMap if present (encodes slit shapes), else radial.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) / (size / 2)         # 0 centre .. 1 edge
    ang = (np.arctan2(dy, dx) + math.pi) / (2 * math.pi)  # 0..1
    pm_path = os.path.join(src, "pupilMap.png")
    dist = _load_gray(pm_path, size) if os.path.exists(pm_path) else r

    # Iris lookup: distance maps to the strip's rows once past the pupil
    d = np.clip((dist - pupil) / max(1e-6, 1.0 - pupil), 0, 0.999)
    iy = (d * (ih - 1)).astype(np.int32)
    ix = (ang * (iw - 1)).astype(np.int32)
    rgb = iris[iy, ix]

    is_pupil = dist < pupil
    rgb[is_pupil] = 0.0
    # soft pupil edge
    edge = np.clip((dist - pupil) / 0.03, 0, 1)[..., None]
    rgb = rgb * edge

    if sclera is not None:
        # iris occupies the inner part; sclera outside (uncanny default eyes)
        iris_r = 0.62
        in_iris = (r <= iris_r)[..., None]
        rgb = np.where(in_iris, rgb, sclera)
        limbus = np.clip((iris_r - r) / 0.05, 0, 1)[..., None]
        rgb = rgb * (0.55 + 0.45 * limbus) if True else rgb

    # Outline: the eye disc, trimmed by the lids at their open position
    visible = (r <= 1.0)
    for lid in ("lid-upper.png", "lid-lower.png"):
        p = os.path.join(src, lid)
        if os.path.exists(p):
            visible &= _load_gray(p, size) >= lid_open
    # darken toward the rim for depth
    rim = np.clip((1.0 - r) / 0.12, 0, 1)[..., None]
    rgb = rgb * (0.35 + 0.65 * rim)

    alpha = visible.astype(np.float32)
    img = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=2)
    out = Image.fromarray((img * 255).astype(np.uint8), "RGBA")
    # soften the outline
    a = out.split()[3].filter(ImageFilter.GaussianBlur(1.2))
    out.putalpha(a)
    if highlight:
        hl = Image.new("RGBA", out.size, (0, 0, 0, 0))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(hl)
        hx, hy, hr = int(cx - size * 0.18), int(cy - size * 0.2), int(size * 0.07)
        dr.ellipse((hx - hr, hy - hr * 0.7, hx + hr, hy + hr * 0.7), fill=(255, 255, 255, 150))
        hl = hl.filter(ImageFilter.GaussianBlur(size * 0.012))
        hl.putalpha(Image.fromarray((np.asarray(hl.split()[3]) * (alpha > 0.5)).astype(np.uint8)))
        out = Image.alpha_composite(out, hl)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="folder with iris.png, pupilMap.png, lid-upper.png, lid-lower.png[, sclera.png]")
    p.add_argument("out", help="face folder to write eye_left.png / eye_right.png into")
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--pupil", type=float, default=0.22, help="pupil size as a fraction of the distance field")
    p.add_argument("--lid-open", type=float, default=0.55, help="lid mask threshold; higher = more almond-shaped")
    p.add_argument("--aspect", type=float, default=1.0, help="width multiplier, e.g. 0.8 makes a narrower eye")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    right = render_eye(args.src, args.size, args.pupil, args.lid_open)
    if args.aspect != 1.0:
        right = right.resize((max(1, int(args.size * args.aspect)), args.size), Image.LANCZOS)
    right.save(os.path.join(args.out, "eye_right.png"))
    right.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(args.out, "eye_left.png"))
    print(f"wrote {args.out}/eye_left.png and eye_right.png ({args.size}x{args.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
