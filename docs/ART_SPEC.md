# Face art spec (locked)

Give this file, unchanged, to an image-generation model or an artist when you
want art for a new Talker face. Every rule below is required by the renderer;
do not relax them. The rendering code is the source of truth
(`talker/face_asset_loader.py`); this document restates its expectations.

## Deliverable

One folder named after the character, containing PNG files from the list
below plus a `face.json` (copy `faces/_template/face.json` and edit). Any file
may be omitted; missing parts fall back to glowing procedural shapes.

| File | Content | Required? |
|------|---------|-----------|
| `face_base.png` | head, hair, body, clothing: everything that does not move | optional |
| `eye_left.png` | the left eye only (viewer's left) | optional |
| `eye_right.png` | the right eye only | optional |
| `nose.png` | nose only | optional |
| `mouth_sil.png` | mouth closed, at rest | recommended |
| `mouth_pp.png` | lips pressed together (p, b, m) | recommended |
| `mouth_aa.png` | wide open (a, ah, aw) | recommended |
| `mouth_oo.png` | tight round (oo, ow) | recommended |
| `mouth_ee.png` | wide flat, teeth showing (ee, ih) | recommended |
| `mouth_ah.png` | relaxed half open (uh, er) | recommended |

Six mouths cover all twelve mouth shapes the renderer uses.

## Canvas

- **`face_base.png` and every `mouth_*.png`: 800 × 800 pixels, PNG, RGBA with
  a transparent background, the full canvas.** Position the part where it sits
  on the finished face; the renderer overlays these files at (0, 0). A cropped
  export renders in the wrong place.
- **Eyes and nose may be standalone images of any size** (the eye alone on a
  transparent background). The renderer centres them on the `cx`/`cy` in
  `face.json` and applies `scale`. Full-canvas exports are also accepted.
- Transparent everywhere except the part itself. No background colour, no
  white box, no drop shadow baked in (the renderer adds glow).
- Design for projection on black: shapes should read on a black ground.
  Avoid thin dark outlines on dark areas; they vanish.

## Consistency across files

- All files come from the same composition: same head, same scale, same
  light direction. The eyes in `eye_left.png` must sit exactly where the
  sockets are in `face_base.png`.
- All six mouths share the same anchor point and the same lip colour and
  line weight; only the opening changes. The lower lip moves, the upper lip
  and the corners stay close to where they are in `mouth_sil.png`.
- Eyes are drawn open and level. The renderer squishes them for blinks and
  tilts them for emotion; do not draw eyelids half closed or eyes already
  angled.
- Keep the eye and mouth files free of surrounding skin or fur unless it
  moves with them; otherwise the base shows through at the edges when the
  eye squishes.

## Style

Flat or lightly shaded, clean edges, no photographic textures, no text, no
watermark. Colour is free. A character for a museum or school should be
recognisable but not a photograph of a real person.

## Prompt scaffold for an image model

Use one prompt per file. Keep the character description identical in all of
them and change only the last sentence.

> An 800 by 800 pixel PNG with a fully transparent background, flat vector
> illustration, clean edges, no text. Character: [one sentence: who they are,
> age, key features, colours, mood]. Show ONLY the [face base / left eye /
> right eye / mouth, closed at rest / mouth, lips pressed / mouth wide open /
> mouth tight round / mouth wide flat with teeth / mouth relaxed half open],
> placed exactly where it sits on the face, everything else transparent.

After generation, run `python tools/check_face.py faces/<name>` to validate the
folder (sizes, transparency, alignment of the mouth states), then set
`eye_left.cx/cy`, `eye_right.cx/cy` and `mouth.anchor_cx/cy` in `face.json`
to the centres of those parts and run:

```bash
python speak.py --face <name> --debug --text "Testing one two three"
```

## Fewer images: the textured mouth (planned)

Instead of six mouth states, two images: `mouth_closed.png` (lips at rest)
and `mouth_inside.png` (teeth and tongue, drawn as if the mouth were open).
The renderer derives the opening from the lip outline and animates it
continuously. When this lands, prefer it for generated art: two consistent
images are far easier to produce than six aligned ones.
