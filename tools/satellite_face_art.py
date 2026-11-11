#!/usr/bin/env python3
"""
satellite_face_art.py — a talker face, rendered into the art the satellite firmware draws.

    python tools/satellite_face_art.py faces/clara \
        --out ~/esp/xiaozhi-esp32/main/boards/waveshare/esp32-s3-touch-amoled-1.8-v2 \
        --lvgl-image ~/esp/xiaozhi-esp32/scripts/Image_Converter/LVGLImage.py

What comes out, for a face called <prefix> (the folder name unless --prefix says otherwise):

    <prefix>_left_sclera.c    RGB565A8   the eye at rest without its iris, alpha = the eye disc
    <prefix>_left_iris.c      RGB565A8   the iris and pupil, a disc that moves with the gaze
    <prefix>_left_lid0..4.c   A8         what the lids cover at five openings (4 = shut); also
                                         masks everything outside the eye disc
    <prefix>_left_emo_*.c     A8         the expression's lid cut per emotion (happy, angry,
                                         annoyed, sad), drawn under the blink mask
    (and the same for right)
    <prefix>_mouth_<viseme>.c RGB565     the six mouth shapes on black, cropped to their pixels
    <prefix>_face_layout.h               sizes and positions for clara_face_display.cc

The eye layers come from the same textured-eye model the host face uses (talker/textured_eye.py)
and the same lid geometry (EMOTION_LIDS and _lid_mask in talker/face_asset_loader.py), so the
board shows the character the desktop shows. The mouths are the face's own mouth PNGs.

Geometry: the face's 16:9 canvas is scaled by --scale (0.6 for a 1280x720 face on the 368x448
panel) and centred on the panel; what falls outside is cut. Eye size, iris size, gaze range and
every position are derived from face.json and written to the layout header, so the firmware has
no numbers of its own. The C arrays are made by LVGL's own LVGLImage.py (in the firmware repo).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from talker.textured_eye import TexturedEye, TexturedEyeAssets          # noqa: E402

# upper cover, lower cover, tilt, lower-lid arch, upper-lid corner drop: talker/face_asset_loader.py
EMOTION_LIDS = {
    "happy":   (0.00, 0.18,  0.00, 0.30, 0.30),
    "angry":   (0.40, 0.05,  0.30, 0.00, 0.00),
    "annoyed": (0.30, 0.05,  0.10, 0.00, 0.00),
    "sad":     (0.28, 0.05, -0.30, 0.00, 0.00),
}
# viseme ids on the wire (talker/satellite.py VISEME_WIRE): 0 sil, 1 pp, 2 ee, 3 ah, 4 aa, 5 oo
MOUTH_ORDER = ["sil", "pp", "ee", "ah", "aa", "oo"]
LID_LEVELS = 5


def lid_alpha(eye: TexturedEye, blink: float) -> np.ndarray:
    """The eye's visible area (0..1) at a blink amount: the disc cut by both lid maps.
    Same thresholds as TexturedEye.render."""
    a = eye.a
    u_thr = eye.lid_open + (1.0 - eye.lid_open) * blink
    l_thr = eye.lid_open + (1.0 - eye.lid_open) * blink * 0.6
    alpha = eye._disc.copy()
    if a.lid_upper is not None:
        alpha *= np.clip((a.lid_upper - u_thr) / 0.04 + 0.5, 0, 1)
    if a.lid_lower is not None:
        alpha *= np.clip((a.lid_lower - l_thr) / 0.04 + 0.5, 0, 1)
    return alpha


def emotion_cut(n: int, lids: Tuple[float, ...], is_left: bool, span: Tuple[float, float]) -> np.ndarray:
    """What an expression's lids cover, 0..1, the polygon cut of face_asset_loader._lid_mask."""
    upper, lower, tilt, arch, drop = lids
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    top, bottom = span[0] * n, span[1] * n
    core = max(1.0, bottom - top)
    steps = 24
    xs = [i / steps for i in range(steps + 1)]
    if upper > 0.01 or drop > 0.01:
        pts = [(0, 0), (n, 0)]
        for x in reversed(xs):
            inner = x if is_left else 1.0 - x
            edge = (2 * x - 1) ** 2
            cover = upper + tilt * (inner - 0.5) + drop * edge
            pts.append((int(x * n), int(top + max(0.0, cover) * core)))
        d.polygon(pts, fill=255)
    if lower > 0.01:
        pts = [(0, n), (n, n)]
        for x in reversed(xs):
            bow = arch * (1.0 - (2 * x - 1) ** 2)
            cover = lower + bow
            pts.append((int(x * n), int(bottom - max(0.0, cover) * core)))
        d.polygon(pts, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


def render_eye_layers(face_dir: str, te: dict, n: int, is_left: bool) -> Dict[str, Image.Image]:
    """One eye's layers as PIL images, keyed by the layer name used in the C symbols."""
    folder = os.path.join(face_dir, te.get("dir", "eye"))
    assets = TexturedEyeAssets(folder, n, mirror=is_left)        # the host mirrors the left eye
    eye = TexturedEye(assets, lid_open=float(te.get("lid_open", 0.55)),
                      iris_radius=float(te.get("iris_radius", 0.62)), rim=float(te.get("rim", 1.0)))
    pupil = float(te.get("pupil", [0.12, 0.22, 0.44])[1])
    full = eye.render((0.0, 0.0), pupil, 0.0).astype(np.float32) / 255.0      # (n, n, 4) at rest
    disc = eye._disc
    rr = eye._r

    # sclera: the eye without its iris. Where there is no sclera art the eye is black.
    if assets.sclera is not None:
        sclera_rgb = assets.sclera * eye._rim[..., None]
    else:
        sclera_rgb = np.zeros((n, n, 3), dtype=np.float32)
    sclera = np.empty((n, n, 4), dtype=np.uint8)
    sclera[..., :3] = np.clip(sclera_rgb * 255, 0, 255).astype(np.uint8)
    sclera[..., 3] = (disc * 255).astype(np.uint8)

    # iris: the rendered eye inside the iris radius, cut as a disc with a soft edge
    iris_r = eye.iris_radius
    w = 2 * int(np.ceil(iris_r * n / 2))
    c0 = (n - w) // 2
    crop = full[c0:c0 + w, c0:c0 + w]
    rcrop = rr[c0:c0 + w, c0:c0 + w]
    edge = np.clip((iris_r - rcrop) / 0.02, 0, 1)
    iris = np.empty((w, w, 4), dtype=np.uint8)
    iris[..., :3] = np.clip(crop[..., :3] * 255, 0, 255).astype(np.uint8)
    iris[..., 3] = (edge * 255).astype(np.uint8)

    layers: Dict[str, Image.Image] = {
        "sclera": Image.fromarray(sclera, "RGBA"),
        "iris": Image.fromarray(iris, "RGBA"),
    }
    # lids: what is covered at each opening, outside the disc included. The levels are
    # spaced by how much of the open eye stays visible (all, 3/4, 1/2, 1/4, none), not by
    # the blink amount: the lid maps close most of the eye in the first third of a blink,
    # which would make every level but the first look shut.
    open_alpha = lid_alpha(eye, 0.0)
    open_area = max(1.0, float(open_alpha.sum()))
    for k in range(LID_LEVELS):
        target = 1.0 - k / (LID_LEVELS - 1)
        lo, hi = 0.0, 1.0
        for _ in range(24):                                  # bisect the blink amount
            mid = (lo + hi) / 2
            if lid_alpha(eye, mid).sum() / open_area > target: lo = mid
            else: hi = mid
        blink = 0.0 if k == 0 else 1.0 if k == LID_LEVELS - 1 else (lo + hi) / 2
        covered = 1.0 - lid_alpha(eye, blink)
        layers[f"lid{k}"] = mask_image(covered)
    # emotions: the expression's cut inside the open eye's vertical span
    rows = np.where(open_alpha.max(axis=1) > 0.02)[0]
    span = (rows[0] / n, (rows[-1] + 1) / n) if rows.size else (0.0, 1.0)
    for name, lids in EMOTION_LIDS.items():
        cut = emotion_cut(n, lids, is_left, span)
        layers[f"emo_{name}"] = mask_image(cut)
    return layers, w, c0


def mask_image(covered: np.ndarray) -> Image.Image:
    """An A8 source for LVGLImage.py: black with alpha = covered (it reads the alpha channel)."""
    n = covered.shape[0]
    out = np.zeros((n, n, 4), dtype=np.uint8)
    out[..., 3] = np.clip(covered * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def converter_python() -> str:
    """LVGLImage.py needs pypng (and lz4); the ESP-IDF python env has them, so prefer it."""
    import glob
    for cand in sorted(glob.glob(os.path.expanduser("~/.espressif/python_env/*/bin/python")), reverse=True):
        if subprocess.run([cand, "-c", "import png, lz4"], capture_output=True).returncode == 0:
            return cand
    if subprocess.run([sys.executable, "-c", "import png, lz4"], capture_output=True).returncode == 0:
        return sys.executable
    raise SystemExit("LVGLImage.py needs the pypng and lz4 packages: pip install pypng lz4, "
                     "or run after `source ~/esp/esp-idf/export.sh`")


def convert(lvgl_image: str, png: str, cf: str, out_dir: str) -> str:
    """LVGLImage.py: one PNG to a C array named after the file's stem."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [converter_python(), lvgl_image, "--ofmt", "C", "--cf", cf, "-o", tmp, png]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"LVGLImage.py failed on {png}:\n{r.stdout}\n{r.stderr}")
        stem = os.path.splitext(os.path.basename(png))[0]
        src = os.path.join(tmp, stem + ".c")
        if not os.path.exists(src):
            made = os.listdir(tmp)
            raise SystemExit(f"LVGLImage.py made {made}, not {stem}.c")
        dst = os.path.join(out_dir, stem + ".c")
        shutil.copyfile(src, dst)
        return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("face_dir", help="a face folder with face.json, mouth PNGs and a textured eye")
    ap.add_argument("--out", required=True, help="the firmware board folder the .c files go into")
    ap.add_argument("--lvgl-image", required=True, help="path to LVGLImage.py in the firmware repo")
    ap.add_argument("--prefix", default=None, help="C symbol prefix (default: the face folder name)")
    ap.add_argument("--panel", default="368x448", help="panel size WxH (default 368x448)")
    ap.add_argument("--scale", type=float, default=0.6, help="face canvas to panel scale (default 0.6)")
    ap.add_argument("--png-dir", default=None, help="also keep the intermediate PNGs here")
    args = ap.parse_args()

    face_dir = os.path.abspath(args.face_dir)
    prefix = args.prefix or os.path.basename(face_dir.rstrip("/"))
    pw, ph = (int(v) for v in args.panel.lower().split("x"))
    with open(os.path.join(face_dir, "face.json"), encoding="utf-8") as f:
        face = json.load(f)
    te = face.get("textured_eye")
    if not te:
        raise SystemExit("this face has no textured_eye section; the board art needs one")
    s = args.scale
    W, H = int(face.get("canvas_w", 1280)), int(face.get("canvas_h", 720))
    ox = (W * s - pw) / 2.0                 # the scaled canvas, centred on the panel
    oy = (ph - H * s) / 2.0

    def on_panel(x: float, y: float) -> Tuple[int, int]:
        return int(round(x * s - ox)), int(round(y * s + oy))

    n = int(round(int(te.get("size", 190)) * s))
    gaze_radius = float(te.get("gaze_radius", 0.35))
    png_dir = args.png_dir or tempfile.mkdtemp(prefix="face-art-")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    made: List[str] = []

    # eyes
    iris_w = iris_rest = 0
    for side, is_left in (("left", True), ("right", False)):
        layers, iris_w, iris_rest = render_eye_layers(face_dir, te, n, is_left)
        for name, img in layers.items():
            png = os.path.join(png_dir, f"{prefix}_{side}_{name}.png")
            img.save(png)
            cf = "A8" if name.startswith(("lid", "emo")) else "RGB565A8"
            made.append(convert(args.lvgl_image, png, cf, args.out))

    # mouths: full-canvas PNGs, scaled, cropped to their pixels, on black
    mouth_images = face.get("mouth_images", {})
    mouth_pos: Dict[str, Tuple[int, int, int, int]] = {}
    for key in MOUTH_ORDER:
        fname = mouth_images.get(key)
        if not fname:
            raise SystemExit(f"face.json mouth_images has no '{key}'")
        im = Image.open(os.path.join(face_dir, fname)).convert("RGBA")
        im = im.resize((int(round(im.width * s)), int(round(im.height * s))), Image.LANCZOS)
        bbox = im.getbbox()
        if bbox is None:
            raise SystemExit(f"{fname} is empty")
        crop = im.crop(bbox)
        black = Image.new("RGBA", crop.size, (0, 0, 0, 255))
        black.alpha_composite(crop)
        png = os.path.join(png_dir, f"{prefix}_mouth_{key}.png")
        black.convert("RGB").save(png)
        made.append(convert(args.lvgl_image, png, "RGB565", args.out))
        mouth_pos[key] = (int(round(bbox[0] - ox)), int(round(bbox[1] + oy)), crop.width, crop.height)

    # layout header
    lcx, lcy = on_panel(face["eye_left"]["cx"], face["eye_left"]["cy"])
    rcx, rcy = on_panel(face["eye_right"]["cx"], face["eye_right"]["cy"])
    max_shift = int(n * 0.45)
    pos = ", ".join(f"{{{x}, {y}}}" for x, y, _, _ in (mouth_pos[k] for k in MOUTH_ORDER))
    header = os.path.join(args.out, f"{prefix}_face_layout.h")
    with open(header, "w", encoding="utf-8") as f:
        f.write(f"""// Generated by talker tools/satellite_face_art.py from {os.path.basename(face_dir)}/face.json
// (scale {s}, panel {pw}x{ph}). Do not edit; rerun the tool.
#pragma once
#define FACE_PREFIX {prefix}
#define FACE_SYM2(p, x) p##_##x
#define FACE_SYM1(p, x) FACE_SYM2(p, x)
#define FACE_SYM(x) FACE_SYM1(FACE_PREFIX, x)     // FACE_SYM(mouth_sil) -> {prefix}_mouth_sil
#define FACE_EYE_SIZE {n}
#define FACE_IRIS_SIZE {iris_w}
#define FACE_IRIS_REST {iris_rest}
#define FACE_MAX_SHIFT {max_shift}
#define FACE_GAZE_RADIUS {gaze_radius}f
#define FACE_EYE_L_CX {lcx}
#define FACE_EYE_L_CY {lcy}
#define FACE_EYE_R_CX {rcx}
#define FACE_EYE_R_CY {rcy}
// one {{x, y}} per viseme on the wire: sil, pp, ee, ah, aa, oo
#define FACE_MOUTH_POS {{{pos}}}
""")
    print(f"{len(made)} images and {os.path.basename(header)} -> {args.out}")
    print(f"eye {n} px, iris {iris_w} px at {iris_rest}, eyes at ({lcx},{lcy}) ({rcx},{rcy}), "
          f"mouths " + " ".join(f"{k}:{w}x{h}@{x},{y}" for k, (x, y, w, h) in mouth_pos.items()))
    if args.png_dir is None:
        shutil.rmtree(png_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
