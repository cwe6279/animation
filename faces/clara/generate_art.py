"""
Generates Clara's art: two eyes and six mouth shapes, no head and no visor.

Same treatment as EVE's eyes (soft oval, dark rim to light core, a halo around
it) applied to the mouth as well, so the mouth reads as the same kind of light
rather than a drawn lip. Everything floats on black, which is what a projector
wants.

Run once (python faces/ada/generate_art.py); edit the numbers at the top and
rerun to reshape her. Needs Pillow.

The canvas is 16:9 so the face fills a projector or an HDMI display edge to
edge, with no black bars at the sides.
"""
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
SS = 4                       # supersampling, for smooth edges

RIM = (28, 92, 225)          # outer colour of every glowing shape
MID = (60, 150, 255)
CORE = (140, 210, 255)       # centre, brightest

EYE_CY = 300                 # both eyes sit on this line
EYE_L_CX, EYE_R_CX = 520, 760
EYE_RX, EYE_RY = 70, 34      # half width and half height of an eye
EYE_TILT = 7                 # outer corners lifted, like EVE's

MOUTH_CX, MOUTH_CY = 640, 452

# viseme key -> (half width, half height). Six images cover all twelve shapes.
MOUTHS = {
    "sil": (42, 6),          # closed: a thin line of light
    "pp":  (34, 5),          # pressed lips
    "ee":  (58, 12),         # wide and flat
    "ah":  (46, 27),         # neutral open
    "aa":  (52, 39),         # wide open
    "oo":  (27, 27),         # tight and round
}


def glow_oval(cx, cy, rx, ry, tilt_deg=0.0):
    """A soft oval of light: rim colour outside, core colour in the middle, halo around."""
    s = SS
    img = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    core = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(core)
    steps = 14
    for i in range(steps):
        f = i / (steps - 1)
        col = tuple(int(RIM[k] + (CORE[k] - RIM[k]) * f ** 1.4) for k in range(3))
        ax, ay = rx * s * (1 - 0.8 * f), ry * s * (1 - 0.8 * f)
        d.ellipse((cx * s - ax, cy * s - ay, cx * s + ax, cy * s + ay), fill=(*col, 255))
    core = core.filter(ImageFilter.GaussianBlur(3 * s))
    halo = core.filter(ImageFilter.GaussianBlur(18 * s))
    halo.putalpha(halo.split()[3].point(lambda a: int(a * 0.55)))
    img = Image.alpha_composite(img, halo)
    img = Image.alpha_composite(img, core)
    if tilt_deg:
        img = img.rotate(tilt_deg, resample=Image.BICUBIC, center=(cx * s, cy * s))
    return img.resize((W, H), Image.LANCZOS)


# ── live eye parts (talker/textured_eye.py) ──────────────────────────────────
# Her eye is the cat's, which is Adafruit's Uncanny Eyes "dragon" design: the same
# lids, sclera and iris texture, with the iris recoloured from green to ice blue and
# the slit pupil replaced by a round one. Only the colour scale and the pupil change,
# so the fibre detail of the original survives.
TEMPLATE_EYE = os.path.join(os.path.dirname(HERE), "cat", "eye")
EYE_N = 160                  # the round pupil map and lid masks are square, this many pixels
LID_OPENING = 0.32           # half-height of the open eye: raise it to show more iris
LID_WIDTH = 1.0              # half-width: the corners meet at the outline
LID_TAPER = 0.75             # 0.5 is a plain ellipse with blunt ends; higher tapers the
                             # corners to almond points while keeping the middle full
LID_THRESHOLD = 0.55         # must match "lid_open" in face.json

# luminance of the template maps onto this ramp: dark rim, ice blue body, white core
ICE_RAMP = [(0.00, (6, 20, 44)), (0.35, (26, 96, 170)), (0.65, (110, 198, 242)),
            (1.00, (234, 250, 255))]


def recolour_iris(path):
    """Same texture, new colour scale: map each pixel's brightness onto the ice ramp."""
    import numpy as np
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)         # keep the detail
    lum = (lum - lum.min()) / max(1e-6, float(lum.max() - lum.min()))   # use the full ramp
    stops = np.array([s for s, _ in ICE_RAMP], dtype=np.float32)
    cols = np.array([c for _, c in ICE_RAMP], dtype=np.float32)
    out = np.stack([np.interp(lum, stops, cols[:, k]) for k in range(3)], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGB")


def pupil_map():
    """Round pupil: a plain radial distance field, dark at the centre."""
    import numpy as np
    g = (np.arange(EYE_N) + 0.5) / EYE_N * 2 - 1
    r = np.hypot(*np.meshgrid(g, g, indexing="xy"))
    return Image.fromarray((np.clip(r, 0, 1) * 255).astype("uint8"), "L")


def lid(upper: bool):
    """Grey mask: a pixel shows while its value is above the lid threshold.

    The open eye is an almond: full through the middle, tapering to corners where the
    two lids meet, rather than the cat's pointed feline shape. LID_OPENING is its
    half-height, so raising it shows more iris, and LID_TAPER sets how sharply the
    corners close. The field is built so its contour at LID_THRESHOLD is exactly that
    outline, and a rising threshold sweeps the lid across it, which is what a blink does.
    """
    import numpy as np
    g = (np.arange(EYE_N) + 0.5) / EYE_N
    x, y = np.meshgrid(g, g, indexing="xy")
    u = np.clip((2 * x - 1) / LID_WIDTH, -1, 1)
    half = LID_OPENING * np.maximum(0.0, 1 - u * u) ** LID_TAPER    # almond half-height at x
    edge = 0.5 - half if upper else 0.5 + half
    v = (y - edge if upper else edge - y) + LID_THRESHOLD
    return Image.fromarray((np.clip(v, 0, 1) * 255).astype("uint8"), "L")


def write_eye_parts():
    import shutil
    out = os.path.join(HERE, "eye")
    os.makedirs(out, exist_ok=True)
    recolour_iris(os.path.join(TEMPLATE_EYE, "iris.png")).save(os.path.join(out, "iris.png"))
    pupil_map().save(os.path.join(out, "pupilMap.png"))       # round, not the template's slit
    lid(True).save(os.path.join(out, "lid-upper.png"))        # rounder than the template's
    lid(False).save(os.path.join(out, "lid-lower.png"))
    shutil.copy(os.path.join(TEMPLATE_EYE, "sclera.png"), os.path.join(out, "sclera.png"))
    print(f"wrote an ice-blue iris, a round pupil map and the template's lids in {out}")


if __name__ == "__main__":
    # Eyes are drawn on the canvas centre and placed by cx/cy in face.json, like EVE's.
    glow_oval(W // 2, H // 2, EYE_RX, EYE_RY, -EYE_TILT).save(os.path.join(HERE, "eye_left.png"))
    glow_oval(W // 2, H // 2, EYE_RX, EYE_RY, +EYE_TILT).save(os.path.join(HERE, "eye_right.png"))
    # Mouths keep their real place on the canvas, so every shape opens from the same centre.
    for key, (rx, ry) in MOUTHS.items():
        glow_oval(MOUTH_CX, MOUTH_CY, rx, ry).save(os.path.join(HERE, f"mouth_{key}.png"))
    print(f"wrote eye_left.png, eye_right.png and {len(MOUTHS)} mouth_*.png in {HERE}")
    write_eye_parts()
