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
EYE_L_CX, EYE_R_CX = 540, 740
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


if __name__ == "__main__":
    # Eyes are drawn on the canvas centre and placed by cx/cy in face.json, like EVE's.
    glow_oval(W // 2, H // 2, EYE_RX, EYE_RY, -EYE_TILT).save(os.path.join(HERE, "eye_left.png"))
    glow_oval(W // 2, H // 2, EYE_RX, EYE_RY, +EYE_TILT).save(os.path.join(HERE, "eye_right.png"))
    # Mouths keep their real place on the canvas, so every shape opens from the same centre.
    for key, (rx, ry) in MOUTHS.items():
        glow_oval(MOUTH_CX, MOUTH_CY, rx, ry).save(os.path.join(HERE, f"mouth_{key}.png"))
    print(f"wrote eye_left.png, eye_right.png and {len(MOUTHS)} mouth_*.png in {HERE}")
