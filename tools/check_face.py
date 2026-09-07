"""
tools/check_face.py — validate a face folder against docs/ART_SPEC.md.

    python tools/check_face.py faces/cat
    python tools/check_face.py path/to/new_face --json      # machine-readable, for an asset-generation loop

Checks the deterministic half of "does this face have the assets it needs":
files present, canvas size and transparency, opaque bounds sensible, eyes
placed where face.json says, mouth states sharing an anchor. Prints what is
missing or wrong and exits 1 if anything blocks rendering. Pillow is the
only extra dependency.
"""

from __future__ import annotations

import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

MOUTH_KEYS = ["sil", "pp", "aa", "oo", "ee", "ah"]
OPTIONAL_PARTS = ["face_base", "eye_left", "eye_right", "nose"]


@dataclass
class Finding:
    level: str          # "error" | "warn" | "info"
    file: str
    message: str


@dataclass
class Report:
    face: str
    findings: List[Finding] = field(default_factory=list)
    present: List[str] = field(default_factory=list)
    missing_recommended: List[str] = field(default_factory=list)

    def add(self, level: str, file: str, msg: str) -> None:
        self.findings.append(Finding(level, file, msg))

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)


def _bounds(img) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of non-transparent pixels, or None if fully transparent."""
    if img.mode != "RGBA":
        return None
    return img.split()[3].getbbox()


def check_image(path: str, report: Report, canvas: Tuple[int, int],
                full_canvas: bool) -> Optional[Tuple[int, int, int, int]]:
    """
    full_canvas=True  (face_base, mouth states): must be exactly the canvas size,
                      because they are laid over the face at (0, 0).
    full_canvas=False (eyes, nose): any size; the renderer centres them on cx/cy
                      and applies `scale`, so a standalone eye image is fine.
    """
    import numpy as np
    from PIL import Image
    name = os.path.basename(path)
    try:
        img = Image.open(path)
    except Exception as e:
        report.add("error", name, f"cannot open: {e}")
        return None
    if full_canvas and img.size != canvas:
        report.add("error", name, f"is {img.size[0]}x{img.size[1]}, must be {canvas[0]}x{canvas[1]} "
                                  "(full canvas: it is laid over the face at the top-left)")
    if img.mode != "RGBA":
        report.add("error", name, f"mode {img.mode}, must be RGBA (transparent background)")
        return None
    box = _bounds(img)
    if box is None:
        report.add("error", name, "is fully transparent")
        return None
    w, h = box[2] - box[0], box[3] - box[1]
    alpha = np.asarray(img.split()[3])
    coverage = float((alpha > 0).mean())
    if coverage > 0.985 and full_canvas:
        report.add("error", name, "has no transparency (background filled) — export with a transparent background")
    if full_canvas and box[0] == 0 and box[1] == 0 and (w < img.size[0] * 0.5 and h < img.size[1] * 0.5):
        report.add("warn", name, "opaque pixels start at the top-left corner: looks like a cropped export placed at (0,0)")
    if not full_canvas and max(img.size) > 2048:
        report.add("warn", name, f"{img.size[0]}x{img.size[1]} is large; set `scale` in face.json or export smaller")
    return box


def check_face(face_dir: str) -> Report:
    report = Report(face=os.path.basename(os.path.abspath(face_dir)))
    manifest_path = os.path.join(face_dir, "face.json")
    if not os.path.isfile(manifest_path):
        report.add("error", "face.json", "missing (copy faces/_template/face.json)")
        return report
    try:
        m = json.load(open(manifest_path, encoding="utf-8"))
    except Exception as e:
        report.add("error", "face.json", f"invalid JSON: {e}")
        return report
    canvas = (int(m.get("canvas_w", 800)), int(m.get("canvas_h", 800)))

    if not os.path.isfile(os.path.join(face_dir, "character.md")):
        report.add("warn", "character.md", "missing: the face will have no personality of its own")

    centers = {}
    # Named parts referenced from the manifest
    refs = {
        "face_base": m.get("face_base"),
        "eye_left": (m.get("eye_left") or {}).get("image"),
        "eye_right": (m.get("eye_right") or {}).get("image"),
        "nose": (m.get("nose") or {}).get("image"),
    }
    for part, fname in refs.items():
        if not fname:
            report.add("info", part, "not set: procedural fallback will be used" if part != "nose" and part != "face_base"
                       else "not set (optional)")
            continue
        path = os.path.join(face_dir, fname)
        if not os.path.isfile(path):
            report.add("error", fname, f"referenced by face.json as {part} but not found")
            continue
        report.present.append(fname)
        box = check_image(path, report, canvas, full_canvas=(part == "face_base"))
        if box:
            centers[part] = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box)

    # Eyes: placement vs face.json, symmetry, size
    for side in ("eye_left", "eye_right"):
        if side in centers:
            cx, cy, box = centers[side]
            ec = m.get(side, {})
            want = (float(ec.get("cx", 0)), float(ec.get("cy", 0)))
            scale = float(ec.get("scale", 1.0))
            w, h = box[2] - box[0], box[3] - box[1]
            if w * scale < 20 or h * scale < 12:
                report.add("warn", refs[side], f"very small after scale ({w*scale:.0f}x{h*scale:.0f} px)")
    if "eye_left" in centers and "eye_right" in centers:
        bl, br = centers["eye_left"][2], centers["eye_right"][2]
        wl, hl = bl[2] - bl[0], bl[3] - bl[1]
        wr, hr = br[2] - br[0], br[3] - br[1]
        if abs(wl - wr) > 0.25 * max(wl, wr) or abs(hl - hr) > 0.25 * max(hl, hr):
            report.add("warn", "eyes", f"left ({wl}x{hl}) and right ({wr}x{hr}) differ in size by more than 25%")

    # Mouths: presence and shared anchor
    mouth_images = {str(k).lower(): v for k, v in (m.get("mouth_images") or {}).items()}
    anchors = []
    for key in MOUTH_KEYS:
        fname = mouth_images.get(key)
        if not fname:
            report.missing_recommended.append(f"mouth_{key}.png")
            continue
        path = os.path.join(face_dir, fname)
        if not os.path.isfile(path):
            report.add("error", fname, f"referenced as mouth {key} but not found")
            continue
        report.present.append(fname)
        box = check_image(path, report, canvas, full_canvas=True)
        if box:
            anchors.append((key, (box[0] + box[2]) / 2, box[1]))
    if mouth_images and not anchors:
        report.add("error", "mouth", "mouth images are listed but none could be read")
    if len(anchors) >= 2:
        xs = [a[1] for a in anchors]
        tops = [a[2] for a in anchors]
        if max(xs) - min(xs) > 20:
            report.add("warn", "mouth", f"mouth states are not horizontally aligned (centres span {max(xs)-min(xs):.0f} px)")
        if max(tops) - min(tops) > 60:
            report.add("warn", "mouth", f"upper lip position varies {max(tops)-min(tops):.0f} px across states; the upper lip should stay put")
    if not mouth_images:
        report.add("info", "mouth", "no mouth images: procedural mouth will be used")
    elif report.missing_recommended:
        report.add("warn", "mouth", f"missing states {', '.join(report.missing_recommended)}: nearest available state will be used")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a face folder against docs/ART_SPEC.md")
    p.add_argument("face_dir")
    p.add_argument("--json", action="store_true", help="print a JSON report (for automation)")
    args = p.parse_args()
    r = check_face(args.face_dir)
    if args.json:
        print(json.dumps({"face": r.face, "ok": r.ok, "present": r.present,
                          "missing_recommended": r.missing_recommended,
                          "findings": [asdict(f) for f in r.findings]}, indent=2))
    else:
        print(f"{r.face}: {'OK' if r.ok else 'PROBLEMS'}  ({len(r.present)} art files)")
        for f in r.findings:
            print(f"  [{f.level:5s}] {f.file}: {f.message}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
