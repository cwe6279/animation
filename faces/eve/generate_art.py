"""
Generates EVE (WALL-E) head art: face_base.png, eye_left.png, eye_right.png.
Run once (python faces/eve/generate_art.py); edit the numbers and rerun to tweak.
Needs Pillow.  All PNGs are full 800x800 canvases, like hand-made art.

Proportions follow the film design: a wide flattened dome (wider than tall,
flatter underneath), a large black visor across the lower front, and two
soft blue ovals with a darker rim and a lighter center.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 800
SS = 4  # supersampling for smooth edges

WHITE = (246, 248, 251, 255)
SHADE = (196, 203, 214, 255)
VISOR = (10, 11, 14, 255)
EYE_RIM = (28, 92, 225)
EYE_MID = (60, 150, 255)
EYE_CORE = (140, 210, 255)

CX, CY = 400, 400           # head center on the canvas
HEAD_A = 262                # half width
HEAD_TOP = 222              # radius upward
HEAD_BOT = 165              # radius downward (flatter underneath)


def dome_points(cx, cy, a, top, bot, n=360, scale=1.0):
    """Wide dome: ellipse with different vertical radii above and below center."""
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = a * math.cos(t) * scale
        r = top if math.sin(t) < 0 else bot
        y = r * math.sin(t) * scale
        pts.append((cx + x, cy + y))
    return pts


def _down(img):
    return img.resize((W, H), Image.LANCZOS)


def face_base():
    S = SS
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    cx, cy = CX * S, CY * S
    # Shell in shade colour, then a slightly smaller white dome offset up-left,
    # blurred so the lower-right rim keeps a soft shadow.
    ImageDraw.Draw(img).polygon(dome_points(cx, cy, HEAD_A * S, HEAD_TOP * S, HEAD_BOT * S), fill=SHADE)
    top = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(top).polygon(
        dome_points(cx - 8 * S, cy - 8 * S, (HEAD_A - 10) * S, (HEAD_TOP - 10) * S, (HEAD_BOT - 12) * S),
        fill=WHITE)
    top = top.filter(ImageFilter.GaussianBlur(7 * S))
    img = Image.alpha_composite(img, top)

    # Visor: rounded rectangle, then clipped to a dome slightly inside the shell
    # so its bottom follows the head's curve.
    visor = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(visor).rounded_rectangle(
        (cx - 200 * S, cy - 95 * S, cx + 200 * S, cy + 175 * S), radius=95 * S, fill=VISOR)
    clip = Image.new("L", img.size, 0)
    ImageDraw.Draw(clip).polygon(
        dome_points(cx, cy, (HEAD_A - 22) * S, (HEAD_TOP - 22) * S, (HEAD_BOT - 16) * S), fill=255)
    visor.putalpha(Image.composite(visor.split()[3], Image.new("L", img.size, 0), clip))
    img = Image.alpha_composite(img, visor)

    # Faint reflection band across the top of the visor
    refl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(refl).ellipse((cx - 170 * S, cy - 92 * S, cx + 170 * S, cy - 30 * S), fill=(120, 170, 255, 34))
    refl = refl.filter(ImageFilter.GaussianBlur(10 * S))
    refl.putalpha(Image.composite(refl.split()[3], Image.new("L", img.size, 0), visor.split()[3]))
    img = Image.alpha_composite(img, refl)
    return _down(img)


def eye(tilt_deg):
    """Soft blue oval (dark rim -> light center) with a halo, centered on the canvas."""
    S = SS
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    cx, cy = W * S // 2, H * S // 2
    rx, ry = 62 * S, 30 * S
    # Radial-ish gradient: stack shrinking ellipses from rim colour to core colour
    core = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(core)
    steps = 14
    for i in range(steps):
        f = i / (steps - 1)
        col = tuple(int(EYE_RIM[k] + (EYE_CORE[k] - EYE_RIM[k]) * f ** 1.4) for k in range(3))
        d.ellipse((cx - rx * (1 - 0.8 * f), cy - ry * (1 - 0.8 * f),
                   cx + rx * (1 - 0.8 * f), cy + ry * (1 - 0.8 * f)), fill=(*col, 255))
    core = core.filter(ImageFilter.GaussianBlur(3 * S))
    halo = core.filter(ImageFilter.GaussianBlur(18 * S))
    halo.putalpha(halo.split()[3].point(lambda a: int(a * 0.55)))
    img = Image.alpha_composite(img, halo)
    img = Image.alpha_composite(img, core)
    return _down(img.rotate(tilt_deg, resample=Image.BICUBIC, center=(cx, cy)))


if __name__ == "__main__":
    face_base().save(os.path.join(HERE, "face_base.png"))
    eye(-7).save(os.path.join(HERE, "eye_left.png"))     # outer (left) corner slightly up
    eye(+7).save(os.path.join(HERE, "eye_right.png"))
    print("wrote face_base.png, eye_left.png, eye_right.png")
