"""
unwrap_eye.py — turn a flat eye picture into live textured-eye parts.

    python faces/tools/unwrap_eye.py faces/cat/eye_right.png faces/cat/eye --size 256

Takes one eye image (RGBA, the eye alone on a transparent background, pupil
roughly in the middle) and writes the parts textured_eye.py animates:

    iris.png        polar strip of the iris, angle across, distance down,
                    sampled outward from the pupil edge (reflections removed)
    pupilMap.png    distance field shaped like the original pupil, so a slit
                    stays a slit when it dilates
    lid-upper.png   threshold masks that reproduce the eye's outline at
    lid-lower.png   lid_open = 0.55 and close it during blinks
    highlight.png   the bright reflections, kept as a fixed overlay
    sclera.png      black (this eye is iris to the edge); edit if yours has a white

Then add to face.json:
    "textured_eye": {"dir": "eye", "size": 256, "lid_open": 0.55, "rim": 0.0,
                     "pupil": [<min>, <base>, <max>], ...}
The script prints the suggested block with the measured pupil size.
Right eye in, left eye is mirrored by the renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from PIL import Image


def _largest_blob(mask: np.ndarray, near: tuple) -> np.ndarray:
    """Connected component of `mask` closest to `near` (flood fill, no scipy)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                cur += 1
                stack = [(y, x)]
                labels[y, x] = cur
                pts = []
                while stack:
                    cy, cx = stack.pop()
                    pts.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
                comps.append((cur, pts))
    if not comps:
        return mask
    def score(c):
        _, pts = c
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        d = math.hypot(ys.mean() - near[0], xs.mean() - near[1])
        return d - 0.02 * len(pts) ** 0.5 * 10   # prefer big blobs near the centre
    best = min(comps, key=score)[0]
    return labels == best


def unwrap(path: str, out_dir: str, size: int = 256, angles: int = 512, radii: int = 80,
           dark: int = 48, bright: int = 205, work: int = 384, pupil_map_from: str = None) -> dict:
    img = Image.open(path).convert("RGBA")
    # work on a downsized copy for the blob search, full-res for sampling
    full = np.asarray(img).astype(np.float32) / 255.0
    H, W = full.shape[:2]
    alpha_full = full[..., 3]
    ys, xs = np.where(alpha_full > 0.5)
    if ys.size == 0:
        raise SystemExit("image is fully transparent")
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()

    # ── find the pupil: darkest blob near the middle of the opaque area ──
    scale = work / max(H, W)
    small = np.asarray(img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR)).astype(np.float32) / 255.0
    lum = small[..., :3].mean(axis=2)
    core = small[..., 3] > 0.5
    # interior only: erode the opaque mask so the dark rim around the eye is excluded
    interior = core.copy()
    for _ in range(max(2, int(work * 0.06))):
        interior = interior & np.roll(interior, 1, 0) & np.roll(interior, -1, 0) & np.roll(interior, 1, 1) & np.roll(interior, -1, 1)
    dark_mask = interior & (lum < dark / 255.0)
    # erode away thin dark strands by requiring dark neighbours
    er = dark_mask.copy()
    for _ in range(3):
        er = er & np.roll(er, 1, 0) & np.roll(er, -1, 0) & np.roll(er, 1, 1) & np.roll(er, -1, 1)
    cy0 = (top + bottom) / 2 * scale; cx0 = (left + right) / 2 * scale
    pupil_small = _largest_blob(er, (cy0, cx0))
    pys, pxs = np.where(pupil_small)
    pcy, pcx = pys.mean() / scale, pxs.mean() / scale          # pupil centre, full-res
    # pupil ellipse from second moments
    cov = np.cov(np.vstack([pxs, pys]) / scale)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    a_axis, b_axis = 4.0 * math.sqrt(max(evals[0], 1e-6)), 4.0 * math.sqrt(max(evals[1], 1e-6))
    theta = math.atan2(evecs[1, 0], evecs[0, 0])

    # ── canvas: square centred on the pupil, big enough to hold the eye ──
    R = max(pcx - left, right - pcx, pcy - top, bottom - pcy) * 1.04
    n = size
    def to_canvas(y, x):            # full-res -> canvas coords
        return (y - pcy) / R * (n / 2) + (n - 1) / 2, (x - pcx) / R * (n / 2) + (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # canvas -> full-res sample coords
    sy = (yy - (n - 1) / 2) / (n / 2) * R + pcy
    sx = (xx - (n - 1) / 2) / (n / 2) * R + pcx
    def sample(arr, y, x):
        yi = np.clip(np.round(y).astype(int), 0, H - 1); xi = np.clip(np.round(x).astype(int), 0, W - 1)
        return arr[yi, xi]
    rgba = sample(full, sy, sx)
    alpha = rgba[..., 3]

    # ── highlights: bright, low-saturation pixels inside the eye ──
    rgb = rgba[..., :3]
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    hl = (alpha > 0.5) & (mx > bright / 255.0) & ((mx - mn) < 0.22)
    highlight = np.zeros((n, n, 4), dtype=np.float32)
    highlight[..., :3] = rgb
    highlight[..., 3] = np.where(hl, np.clip((mx - bright / 255.0) / (1 - bright / 255.0) * 1.2, 0, 0.85), 0)

    # ── pupil distance field on the canvas ──
    dy, dx = yy - (n - 1) / 2, xx - (n - 1) / 2
    ct, st = math.cos(theta), math.sin(theta)
    u = (dx * ct + dy * st) / (a_axis / R * (n / 2) / 2)      # along the pupil's long axis
    v = (-dx * st + dy * ct) / (b_axis / R * (n / 2) / 2)
    r_e = np.sqrt(u * u + v * v)                              # 1 at the fitted pupil boundary
    r_c = np.sqrt(dx * dx + dy * dy) / (n / 2)                # 1 at the canvas circle
    pupil_base = 0.22
    # where the pupil boundary sits in circular terms, per direction
    ang = np.arctan2(dy, dx)
    r_c_at_pupil = np.clip(r_c / np.maximum(r_e, 1e-6), 0, 0.95)
    t = np.clip((r_c - r_c_at_pupil) / np.maximum(1 - r_c_at_pupil, 1e-6), 0, 1)
    dist = np.where(r_e < 1.0, pupil_base * r_e, pupil_base + (1 - pupil_base) * t)
    pupil_map = np.clip(dist, 0, 1)

    # ── iris strip: sample along rays from the pupil edge to the eye outline ──
    strip = np.zeros((radii, angles, 3), dtype=np.float32)
    refl = np.zeros((radii, angles), dtype=bool)          # cells that were reflections
    for ai in range(angles):
        a = 2 * math.pi * ai / angles - math.pi           # renderer's angle convention
        ca, sa = math.cos(a), math.sin(a)
        # pupil boundary radius along this ray (ellipse), in full-res pixels
        ue = ca * ct + sa * st; ve = -ca * st + sa * ct
        r0 = 1.0 / math.sqrt((ue / (a_axis / 2)) ** 2 + (ve / (b_axis / 2)) ** 2 + 1e-9)
        # walk out of the actual pupil (it may be pointier than the fitted ellipse)
        r = max(2.0, r0 * 0.6)
        while r < R:
            y, x = int(np.clip(pcy + sa * r, 0, H - 1)), int(np.clip(pcx + ca * r, 0, W - 1))
            if full[y, x, :3].mean() > dark / 255.0 * 1.4 or alpha_full[y, x] < 0.5:
                break
            r += 1.0
        r0 = r
        # outline radius: walk out until alpha drops, then back off the dark rim
        r1 = r0
        for r in np.arange(r0, R * 1.2, 1.0):
            y, x = pcy + sa * r, pcx + ca * r
            if not (0 <= y < H and 0 <= x < W) or alpha_full[int(y), int(x)] < 0.5:
                break
            r1 = r
        rr = r1
        while rr > r0 + 4 and full[int(np.clip(pcy + sa * rr, 0, H - 1)), int(np.clip(pcx + ca * rr, 0, W - 1)), :3].mean() < dark / 255.0 * 1.6:
            rr -= 1.0
        r1 = max(r0 + 4, rr)
        last_good = None
        pending = []                                    # reflection samples before any iris colour
        for ri in range(radii):
            r = r0 + (r1 - r0) * (ri + 0.5) / radii
            y, x = int(np.clip(pcy + sa * r, 0, H - 1)), int(np.clip(pcx + ca * r, 0, W - 1))
            px = full[y, x, :3]
            bright_px = px.max() > (bright - 30) / 255.0 and (px.max() - px.min()) < 0.3
            if bright_px:
                refl[ri, ai] = True
                if last_good is not None:
                    strip[ri, ai] = last_good           # placeholder; inpainted across angles below
                else:
                    pending.append(ri)
                continue
            last_good = px
            strip[ri, ai] = px
            for pi in pending:                          # backfill the start of the ray
                strip[pi, ai] = px
            pending = []
        if last_good is None:                            # a ray with nothing but reflection
            refl[:, ai] = True
    # inpaint reflection cells from the nearest clean angles (the strip wraps around)
    for ri in range(radii):
        bad = refl[ri]
        if bad.any() and not bad.all():
            good_idx = np.where(~bad)[0]
            for ai in np.where(bad)[0]:
                # nearest clean angle on each side, circularly, then blend
                d = (good_idx - ai) % angles
                right_i = good_idx[np.argmin(d)]
                d2 = (ai - good_idx) % angles
                left_i = good_idx[np.argmin(d2)]
                wr, wl = d.min(), d2.min()
                strip[ri, ai] = (strip[ri, left_i] * wr + strip[ri, right_i] * wl) / max(1, wr + wl)
    # smooth the strip a little along the angle axis to hide sampling noise
    strip = (strip + np.roll(strip, 1, 1) + np.roll(strip, -1, 1)) / 3.0

    # ── lid masks reproducing the outline at threshold 0.55 ──
    lid_open = 0.55
    top_edge = np.full(n, np.nan); bot_edge = np.full(n, np.nan)
    inside = alpha > 0.5
    for x in range(n):
        col = np.where(inside[:, x])[0]
        if col.size:
            top_edge[x], bot_edge[x] = col[0], col[-1]
    # fill columns outside the eye with the nearest known edge
    valid = ~np.isnan(top_edge)
    if valid.any():
        idx = np.arange(n)
        top_edge = np.interp(idx, idx[valid], top_edge[valid])
        bot_edge = np.interp(idx, idx[valid], bot_edge[valid])
    height = np.maximum(bot_edge - top_edge, 1.0)
    upper = np.clip(lid_open + (yy - top_edge[None, :]) / height[None, :] * (1 - lid_open) * 1.6, 0, 1)
    upper = np.where(yy < top_edge[None, :], np.clip(lid_open - (top_edge[None, :] - yy) / (n * 0.15), 0, lid_open - 0.01), upper)
    lower = np.clip(lid_open + (bot_edge[None, :] - yy) / height[None, :] * (1 - lid_open) * 1.6, 0, 1)
    lower = np.where(yy > bot_edge[None, :], np.clip(lid_open - (yy - bot_edge[None, :]) / (n * 0.15), 0, lid_open - 0.01), lower)
    # columns with no eye at all: fully closed
    upper[:, ~valid] = 0.0; lower[:, ~valid] = 0.0

    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(strip, 0, 1) * 255).astype(np.uint8), "RGB").save(os.path.join(out_dir, "iris.png"))
    if pupil_map_from:
        Image.open(pupil_map_from).convert("L").resize((n, n), Image.BILINEAR).save(os.path.join(out_dir, "pupilMap.png"))
    else:
        Image.fromarray((pupil_map * 255).astype(np.uint8), "L").save(os.path.join(out_dir, "pupilMap.png"))
    Image.fromarray((upper * 255).astype(np.uint8), "L").save(os.path.join(out_dir, "lid-upper.png"))
    Image.fromarray((lower * 255).astype(np.uint8), "L").save(os.path.join(out_dir, "lid-lower.png"))
    Image.fromarray((np.clip(highlight, 0, 1) * 255).astype(np.uint8), "RGBA").save(os.path.join(out_dir, "highlight.png"))
    Image.new("L", (n, n), 0).save(os.path.join(out_dir, "sclera.png"))

    pupil_frac = pupil_base
    block = {"dir": os.path.basename(out_dir.rstrip("/")), "size": n, "lid_open": lid_open, "rim": 0.0,
             "gaze_radius": 0.25, "pupil": [round(pupil_frac * 0.55, 3), pupil_frac, round(min(0.6, pupil_frac * 1.9), 3)],
             "lid_tracking": 0.3}
    info = {"pupil_centre": [round(pcx), round(pcy)], "pupil_axes_px": [round(a_axis), round(b_axis)],
            "pupil_angle_deg": round(math.degrees(theta)), "canvas_radius_px": round(R), "textured_eye": block}
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="right eye PNG (RGBA)")
    p.add_argument("out", help="output folder, e.g. faces/<name>/eye")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--dark", type=int, default=48, help="max brightness (0-255) counted as pupil")
    p.add_argument("--bright", type=int, default=222, help="min brightness counted as a reflection")
    p.add_argument("--pupil-map", default=None,
                   help="borrow another design's pupilMap.png (e.g. faces/dragon/eye/pupilMap.png) instead of fitting one")
    args = p.parse_args()
    info = unwrap(args.image, args.out, size=args.size, dark=args.dark, bright=args.bright, pupil_map_from=args.pupil_map)
    print(f"pupil centre {info['pupil_centre']}, axes {info['pupil_axes_px']} px at {info['pupil_angle_deg']} deg, "
          f"canvas radius {info['canvas_radius_px']} px")
    print("wrote iris.png, pupilMap.png, lid-upper.png, lid-lower.png, highlight.png, sclera.png to", args.out)
    print("add to face.json:\n  \"textured_eye\": " + json.dumps(info["textured_eye"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
