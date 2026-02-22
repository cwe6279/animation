# Talker — Phoneme-Synced Animated Face

A real-time animated talking face driven by TTS audio and phoneme lip sync.
Speak any text and watch the face animate in sync with the words.
Designed for projection on black — faces render as floating features with glow effects.

## Features
- Phoneme-accurate lip sync via edge-tts word timestamps + g2p phoneme mapping
- Multiple faces: **pumpkin**, **green_cat**, **skull**, **ghost** (or create your own)
- Pure black background — ideal for projection mapping
- Artist-friendly: drop in PNG art and tweak `face.json` to position/scale
- Procedural fallback — missing art files render as glowing shapes with drop shadows
- Emotion tags — `[angry]`, `[happy]`, `[sad]`, `[surprise]`, `[annoyed]` change eye expressions mid-sentence
- Per-feature opacity control for eyes, mouth, nose, and face base
- Debug overlay showing viseme, emotion, open amount, FPS in real time

## Project Structure

```
talker/
  talker.py              <- main app + built-in procedural renderers
  phoneme_scheduler.py   <- text -> phonemes -> timed viseme schedule + emotion events
  face_asset_loader.py   <- loads artist PNG faces from directories
  test_emotions.py       <- demo script cycling through all emotion tags
  requirements.txt
  llm_integration/
    system_prompt.md     <- LLM prompt for generating text with emotion tags
  faces/
    _template/           <- copy this to create a new face
    pumpkin/             <- procedural pumpkin (no art files needed)
    green_cat/           <- cat with PNG eye + mouth art
    skull/               <- skull with PNG face base + eye sockets
    ghost/               <- procedural ghost (ready for art)
```

## Setup

### System dependencies
```bash
# Mac
brew install ffmpeg portaudio

# Ubuntu/Debian
sudo apt install ffmpeg portaudio19-dev

# Windows
# ffmpeg: pip install imageio-ffmpeg (bundled, no PATH setup needed)
# PyAudio: pip install pyaudio (works on Python 3.12)
```

### Python packages
```bash
pip install -r requirements.txt
```

Requirements:
- `pygame` — rendering
- `pyaudio` — audio playback
- `numpy` — amplitude analysis
- `edge-tts` — TTS with word timestamps (needs internet)
- `g2p-en` — grapheme-to-phoneme (optional, falls back to built-in regex)
- `imageio-ffmpeg` — bundled ffmpeg for MP3-to-WAV conversion (Windows)

## Usage

```bash
# Pumpkin face with text
python talker.py --face pumpkin --text "Happy Halloween!"

# Green cat face
python talker.py --face green_cat --text "Meow, I am a spooky cat"

# Skull face
python talker.py --face skull --text "I am a talking skull"

# Debug overlay (shows viseme, emotion, open amount, FPS)
python talker.py --face pumpkin --text "Hello" --debug

# Emotion tags inline
python talker.py --face green_cat --text "[angry]I am very angry! [sad]But also sad." --debug

# Default emotion for entire utterance
python talker.py --face green_cat --text "Hello world" --emotion surprise --debug

# Auto-exit (window closes after speech finishes)
python talker.py --face green_cat --text "Quick test" --auto-exit

# Run the emotion demo (cycles through all emotions)
python test_emotions.py

# Play existing WAV file (amplitude mode)
python talker.py --face pumpkin --file my_audio.wav

# Live microphone (amplitude mode)
python talker.py --face pumpkin --mic

# Custom face from any directory
python talker.py --face-dir path/to/my_face --text "Hello"
```

**Keyboard shortcuts in window:**
- `T` — type new text in terminal, speaks it
- `D` — toggle debug overlay
- `ESC` — quit

---

## Emotion Tags

Emotion tags change eye expressions (squish, tilt, blink rate) to convey emotions while speaking.

### Supported Emotions

| Tag | Eyes | Blink rate | Transition speed |
|-----|------|------------|-----------------|
| `[neutral]` | Normal | Normal | Medium |
| `[happy]` | Slight squint | Faster | Medium |
| `[angry]` | Narrowed, tilted inward (V-shape) | Slower | Fast |
| `[annoyed]` | Slightly narrowed, slight tilt | Slightly slower | Medium |
| `[sad]` | Slightly drooped, tilted outward | Faster | Slow |
| `[surprise]` | Wide open | Barely blinks | Fastest |

### Inline Tags

Place `[emotion]` tags anywhere in the text. The emotion applies from that point until the next tag:

```
"[happy]I was having a great day, [angry]but then someone cut me off! [sad]It ruined my mood."
```

Tags are stripped from the text before sending to TTS — they don't affect speech.

### Default Emotion

Use `--emotion` to set a baseline emotion for the entire utterance. Inline tags override it:

```bash
python talker.py --face green_cat --text "Everything is fine." --emotion angry
```

### Emotion Demo

`test_emotions.py` cycles through all emotions with example sentences:

```bash
python test_emotions.py
```

Each demo auto-exits after speech finishes. Press ESC to skip, Ctrl+C to stop all.

### LLM Integration

The [llm_integration/](llm_integration/) folder contains a ready-to-use system prompt (`system_prompt.md`) that teaches an LLM how to use emotion tags correctly. Paste it into your system prompt when building a chatbot or voice agent that outputs text for Talker. It covers tag placement rules, examples, and common mistakes.

---

## Creating Custom Face Art

### Overview

Design your entire face in a single **800x800** document in Adobe Illustrator (or Inkscape, Figma, etc.), then export each feature as a separate PNG layer. The renderer composites them at runtime.

### Step 1: Set Up Your Document

1. Create a new document: **800 x 800 px**, transparent background
2. Design the full face in one place — eyes, mouth, optional nose and face base — all positioned where you want them on the 800x800 canvas
3. Use separate layers for each feature:
   - **Face base** (optional) — head/skull/body silhouette
   - **Left eye**
   - **Right eye**
   - **Nose** (optional)
   - **Mouth states** — 6 variations for lip sync

### Step 2: Export PNGs

Export each layer individually as a **full 800x800 PNG with transparency**:

| File | What to include | Notes |
|------|----------------|-------|
| `face_base.png` | Head/body shape | Optional. Omit for floating features on black. |
| `eye_left.png` | Left eye only | Rest of canvas transparent |
| `eye_right.png` | Right eye only | Rest of canvas transparent |
| `nose.png` | Nose only | Optional |
| `mouth_sil.png` | Mouth closed/neutral | This is the resting state |
| `mouth_pp.png` | Lips pressed (m/b/p) | Bilabial consonants |
| `mouth_aa.png` | Mouth wide open (a/aw) | Widest opening |
| `mouth_oo.png` | Lips rounded (oo/ow) | Tight circle |
| `mouth_ee.png` | Wide flat smile (ee/ih) | Teeth showing |
| `mouth_ah.png` | Relaxed open (uh/er) | Neutral vowel |

**Export by application:**

**Illustrator:**
1. Hide all layers except the one you're exporting
2. File > Export > Export As > PNG, check **"Use Artboards"**
3. Export each layer separately

**Photoshop:**
1. Hide all layers except the one you're exporting
2. File > Export > Export As (or Save for Web)
3. Make sure canvas size stays 800x800 — don't use "Trim" or "Smallest size"

**Krita / GIMP:**
1. Hide all layers except the target
2. Export as PNG — do NOT flatten or autocrop
3. Verify the exported PNG is 800x800

**Key rules:**
- Every PNG **must** be the full 800x800 canvas with the component positioned where it sits on the face
- Use transparency (alpha channel) everywhere except the actual art
- If a PNG is cropped to just the feature, it will render at (0,0) in the top-left corner

### Step 3: Create face.json

Copy `faces/_template/` and rename the folder to your face name. Edit `face.json`:

```json
{
  "name": "my_face",
  "canvas_w": 800,
  "canvas_h": 800,
  "bg_color": [0, 0, 0],

  "face_base": "face_base.png",
  "face_base_opacity": 1.0,

  "glow_color": [255, 160, 0],
  "glow_intensity": 1.0,

  "eye_color": [255, 200, 0],
  "eye_left": {
    "image": "eye_left.png",
    "cx": 300,
    "cy": 300,
    "scale": 1.0,
    "opacity": 1.0
  },
  "eye_right": {
    "image": "eye_right.png",
    "cx": 500,
    "cy": 300,
    "scale": 1.0,
    "opacity": 1.0
  },
  "blink": true,

  "draw_nose": false,
  "nose": {
    "image": "nose.png",
    "cx": 400,
    "cy": 440,
    "scale": 1.0,
    "opacity": 1.0
  },

  "mouth": {
    "anchor_cx": 400,
    "anchor_cy": 520,
    "scale": 1.0,
    "offset_x": 0,
    "offset_y": 0,
    "opacity": 1.0,
    "max_w": 220,
    "min_w": 140,
    "color": [255, 200, 0],
    "dark_color": [10, 10, 10],
    "style": "rounded",
    "n_teeth": 0
  },

  "mouth_images": {
    "sil": "mouth_sil.png",
    "pp":  "mouth_pp.png",
    "aa":  "mouth_aa.png",
    "oo":  "mouth_oo.png",
    "ee":  "mouth_ee.png",
    "ah":  "mouth_ah.png"
  }
}
```

### Step 4: Test and Adjust

```bash
python talker.py --face my_face --text "Testing my new face" --debug
```

Tweak `face.json` without re-exporting PNGs:
- **`cx`, `cy`** — reposition eyes or nose (pixels from top-left)
- **`scale`** — resize eyes, nose, or mouth art (0.5 = half size, 2.0 = double)
- **`opacity`** — transparency per feature (0.0 = invisible, 1.0 = fully opaque)
- **`anchor_cx`, `anchor_cy`** — mouth position (also affects procedural fallback)
- **`offset_x`, `offset_y`** — shift mouth art without changing the anchor
- **`glow_color`** — halo color around procedural shapes
- **`blink`** — enable/disable eye blink animation
- **`draw_nose`** — enable procedural triangle nose (set false if using nose.png or no nose)

### What's Optional

Everything except `face.json` is optional. Missing features fall back to procedural:

| Missing file | Fallback |
|-------------|----------|
| `face_base.png` | No body — floating features on black (projection mode) |
| `eye_*.png` | Glowing triangle eyes (procedural) |
| `nose.png` | Procedural triangle nose (if `draw_nose` is true) |
| Any `mouth_*.png` | Nearest available mouth state, or procedural mouth |
| All `mouth_*.png` | Procedural toothed/rounded mouth using `style` and `color` |

You can incrementally add art — start with just `face.json` for a fully procedural face, then drop in PNGs one at a time.

---

## How Lip Sync Works

```
Text input
  |
edge-tts (boundary="WordBoundary")
  -> WAV audio file
  -> word timestamps: [("Happy", 0.24s, 0.62s), ("Halloween", 0.68s, 1.31s)]
  |
g2p-en (or built-in regex fallback)
  -> ARPAbet phonemes per word: "Happy" -> [HH, AE, P, IY]
  |
Phoneme -> Viseme map (12 shapes)
  -> [AH, AA, PP, EE]
  |
Spread across word duration (vowels get more time than consonants)
  -> VisemeEvent schedule: [{time, viseme, duration}, ...]
  |
PyAudio plays WAV + ScheduleReader checked each frame at 60fps
  |
Renderer: smoothed open/width/rounded -> draws face
```

## The 12 Viseme Shapes

| Viseme | Phonemes       | Mouth shape   | Mouth PNG |
|--------|---------------|---------------|-----------|
| SIL    | silence        | closed        | mouth_sil |
| PP     | p b m          | pressed lips  | mouth_pp  |
| FF     | f v            | lip to teeth  | mouth_ah  |
| TH     | th             | tongue tip    | mouth_ah  |
| DD     | t d n l        | teeth close   | mouth_ah  |
| KK     | k g ng         | back open     | mouth_aa  |
| CH     | ch sh zh j     | puckered      | mouth_oo  |
| SS     | s z            | sibilant      | mouth_ee  |
| AA     | a aw ah        | wide open     | mouth_aa  |
| EE     | ee ih ey       | wide flat     | mouth_ee  |
| OO     | oo ow uh       | tight round   | mouth_oo  |
| AH     | uh er schwa    | neutral       | mouth_ah  |

You only need 6 mouth PNGs — the renderer maps all 12 visemes to the closest available art.

## Troubleshooting

**0 viseme events scheduled**
edge-tts 7.x changed the default boundary to SentenceBoundary. Confirm
`boundary="WordBoundary"` is set in the Communicate() call in phoneme_scheduler.py.

**Mouth stuck on sil / not moving**
Check debug overlay. If Time counts up but Viseme stays sil, the schedule is empty.
If Time is stuck at 0.000, the audio clock is not advancing.

**Mouth PNG in wrong position**
All mouth PNGs must be full 800x800 canvases. If a PNG is cropped to just the mouth
shape, it will render at (0,0) in the top-left corner. Re-export with the full artboard.

**ffmpeg not found**
Install `imageio-ffmpeg` (`pip install imageio-ffmpeg`) for a bundled ffmpeg binary.
Or install ffmpeg system-wide and add to PATH.

**PyAudio install fails on Windows**
Try plain `pip install pyaudio` first (works on Python 3.12).
If that fails, grab the wheel from pypi.org/project/pyaudio/#files
matching cp312 + win_amd64 and install it directly with pip.

**edge-tts fails / no audio**
Needs internet to reach Microsoft speech servers on first call.
