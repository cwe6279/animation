"""
Generates the cyclops's art: one live eye, nothing else.

The eye is Clara's, which is the cat's, which is Adafruit's Uncanny Eyes "dragon"
design. Only the colour scale changes: a brown-red iris instead of ice blue, in a
warm dark surround. The lids are generated here so their shape is a few numbers.

He has one eye and no mouth, so there is no mouth art and the second eye is turned
off in face.json rather than drawn and hidden.

Run: python faces/cyclops/generate_art.py. Needs Pillow.
"""
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ── live eye parts (talker/textured_eye.py) ──────────────────────────────────
# His eye is the cat's, which is Adafruit's Uncanny Eyes "dragon" design: the same
# lids, sclera and iris texture, with the iris recoloured from green to ice blue and
# the slit pupil replaced by a round one. Only the colour scale and the pupil change,
# so the fibre detail of the original survives.
TEMPLATE_EYE = os.path.join(os.path.dirname(HERE), "cat", "eye")
EYE_N = 160                  # the pupil map and lid masks are square, this many pixels
PUPIL_ASPECT = 1.06          # the pupil is round, but a wide almond around it reads as
                             # taller than wide; a touch of extra width cancels that
# luminance of the template maps onto this ramp: near-black rim, brown-red body, a hot
# amber core. One eye in the dark, and it should look like it is lit from behind.
ICE_RAMP = [(0.00, (24, 7, 5)), (0.34, (104, 27, 16)), (0.66, (178, 70, 32)),
            (1.00, (240, 176, 112))]

# Around the iris. The template's is black, which the renderer reads as "no sclera, let the
# iris fill the eye"; a dark one lets iris_radius in face.json shrink the iris while staying
# dark enough on a black screen to read as unlit rather than as a white eyeball.
SCLERA = (28, 15, 11)

# The lids. A real eye is not symmetric: the upper lid sits lower and its peak is
# off-centre toward the nose, the lower lid is shallower, and neither edge is a crisp
# line. These are the knobs; rerun this script after changing any of them.
LID_UPPER_OPEN = 0.30        # half-height of the opening under the upper lid
LID_LOWER_OPEN = 0.35        # ... and above the lower lid, which sits a little further out
LID_WIDTH = 1.0              # half-width: the corners meet at the outline
LID_TAPER_UPPER = 0.88       # the upper lid carries the arch: higher is a deeper curve
LID_TAPER_LOWER = 0.62       # the lower lid is flatter, as a real one is; lower is flatter
LID_PEAK = 0.10              # how far the upper lid's highest point sits off centre
LID_WAVER = 0.022            # a slight irregularity so the edge is not a drawn curve
LID_SOFT = 0.55              # edge slope: lower is a softer, more diffuse lid edge
LID_THRESHOLD = 0.55         # must match "lid_open" in face.json


def recolour_iris(path):
    """Same texture, new colour scale: map each pixel's brightness onto the ramp."""
    import numpy as np
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)         # keep the detail
    lum = (lum - lum.min()) / max(1e-6, float(lum.max() - lum.min()))   # use the full ramp
    stops = np.array([s for s, _ in ICE_RAMP], dtype=np.float32)
    cols = np.array([c for _, c in ICE_RAMP], dtype=np.float32)
    out = np.stack([np.interp(lum, stops, cols[:, k]) for k in range(3)], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGB")


def pupil_map():
    """Round pupil: a radial distance field, dark at the centre. PUPIL_ASPECT widens it
    a little so it looks round inside the almond rather than measuring round."""
    import numpy as np
    g = (np.arange(EYE_N) + 0.5) / EYE_N * 2 - 1
    x, y = np.meshgrid(g, g, indexing="xy")
    r = np.hypot(x / PUPIL_ASPECT, y)
    return Image.fromarray((np.clip(r, 0, 1) * 255).astype("uint8"), "L")


def lid(upper: bool):
    """Grey mask: a pixel shows while its value is above the lid threshold.

    The opening is an almond, full through the middle and tapering to corners where the
    two lids meet. The upper lid carries the arch and its peak sits off centre while the
    lower stays flatter, the edge carries a slight waver, and the field's gentle slope gives the
    renderer a soft edge to fade across instead of a hard line. A rising threshold
    sweeps the lid down over the eye, which is what a blink does.
    """
    import numpy as np
    g = (np.arange(EYE_N) + 0.5) / EYE_N
    x, y = np.meshgrid(g, g, indexing="xy")
    t = 2 * x - 1                                              # -1 at one corner, +1 at the other
    skew = t - (LID_PEAK if upper else -LID_PEAK * 0.4) * (1 - t * t)
    u = np.clip(skew / LID_WIDTH, -1, 1)
    open_to = LID_UPPER_OPEN if upper else LID_LOWER_OPEN
    taper = LID_TAPER_UPPER if upper else LID_TAPER_LOWER
    half = open_to * np.maximum(0.0, 1 - u * u) ** taper
    half = half * (1 + LID_WAVER * (np.sin(4.1 * t + (0.0 if upper else 2.3))
                                    + 0.6 * np.sin(9.7 * t + 1.1)))
    edge = 0.5 - half if upper else 0.5 + half
    v = (y - edge if upper else edge - y) * LID_SOFT + LID_THRESHOLD
    return Image.fromarray((np.clip(v, 0, 1) * 255).astype("uint8"), "L")


def sclera():
    """The dark surround the iris sits in, so the iris can be smaller than the eye."""
    import numpy as np
    a = np.tile(np.array(SCLERA, dtype="uint8"), (EYE_N, EYE_N, 1))
    return Image.fromarray(a, "RGB")


def write_eye_parts():
    import shutil
    out = os.path.join(HERE, "eye")
    os.makedirs(out, exist_ok=True)
    recolour_iris(os.path.join(TEMPLATE_EYE, "iris.png")).save(os.path.join(out, "iris.png"))
    pupil_map().save(os.path.join(out, "pupilMap.png"))       # round, not the template's slit
    lid(True).save(os.path.join(out, "lid-upper.png"))        # rounder than the template's
    lid(False).save(os.path.join(out, "lid-lower.png"))
    sclera().save(os.path.join(out, "sclera.png"))
    print(f"wrote a brown-red iris, a round pupil map and his lids in {out}")


if __name__ == "__main__":
    write_eye_parts()
