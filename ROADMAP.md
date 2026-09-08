# Roadmap

Ideas agreed on but not built yet, roughly in priority order.

## Body (later): servo controls
Ears, eyes, brain and mouth exist; the body is the next organ. A `body.py`
driver with the same shape as the audio and vision pieces:
- **Outputs**: head pan/tilt (or a whole-face turntable), eyelid or ear
  servos, arms or wings, an LED ring. Hardware via a PCA9685 servo board over
  I2C on the Pi, or a serial link to an Arduino; a `NullBody` for desktops.
- **Inputs it listens to**: `EyeMotion` gaze and `look_at` (head follows the
  eyes with lag), emotion changes (ears back for angry, up for surprise),
  speech energy from the audio engine (small head nods on stressed syllables),
  vision notes (turn toward where people are), idle sounds (a shake with a
  bleat).
- **Safety**: rate and range limits per servo in face.json, a soft home
  position on start/stop, and everything off if the loop crashes.
- Config in face.json under `"body"`, off unless present.

## Character sounds (agreed 2026-09-07)
- Each face folder gets a `sounds/` directory of short clips (a bleat, a
  purr, a dragon rumble, a chuckle), generated once (ElevenLabs sound
  effects or v3) or recorded, stored as 24 kHz mono WAV with a matching
  mouth shape / duration in a small `sounds.json`.
- **Intentional**: a tag in the reply (`[bleat]`, `[rumble]`) plays the clip
  through the same audio stream at that point, with the mouth held on its
  shape; the tag is stripped from the voice text.
- **Idle**: while dormant in wake mode (or after a long quiet spell), play
  one at random now and then (configurable interval and probability), with a
  matching glance or blink, so the character feels alive between visitors.
  Optional per face: a `sounds/` folder with `idle.json` listing clips and
  weights; no folder, no sounds.
- **Filler**: on transcript arrival, optionally play a short "thinking" clip
  to cover the first-token wait.

## Canned lines, conversation starters, and filler
- **Pre-rendered phrases.** Synthesize a repertoire once (greetings, "Who goes
  there?", taunts, goodbyes) and cache the PCM *with its word timings* so lip
  sync still works. Playing one is instant: no LLM, no TTS round trip.
- **Conversation starters.** When nobody has spoken for a while, pick a canned
  opener (or ask the LLM for one in the background and cache it) so the face
  initiates instead of waiting.
- **Fillers while thinking.** Between end-of-speech and the first LLM token
  play a short cached beat: a "hmm", a chuckle, an in-character aside. Covers
  the 0.5-1.5 s gap and makes the reply feel immediate.
- **Intercept layer.** If the LLM's first sentence exactly (or fuzzily)
  matches a cached phrase, play the cached audio instead of synthesizing.
- Implementation: a `PhraseCache` keyed by (backend, voice, model, text) ->
  (pcm, word boundaries), populated on first synthesis and by a warm-up
  script; `SpeechPipeline` checks it before calling the backend.

## Latency
- Keep judging brains *inside the loop* with the per-turn `[turn]` line;
  Claude Haiku is the faster Claude when Opus feels slow.
- Pre-open the ElevenLabs websocket for Flash sessions (saves ~100 ms).
- Shorter system prompt variant for the voice loop.

## Raspberry Pi and kiosk deployment
- **Wi-Fi and network setup without a keyboard**: first-boot captive portal or
  a config file on the boot partition (SSID, password, API keys), plus a
  status face state for "no network".
- Cloud speech-to-text by default there (`--stt elevenlabs`).
- Face at 30 fps; measure the renderer on the Pi 5.
- Systemd unit + `--fullscreen` autostart; watchdog that restarts on crash.

## Conversation
- Barge-in without headphones: echo cancellation (WebRTC AEC / PipeWire
  echo-cancel module) so the mic ignores the speaker.
- Wake word or push-to-talk button for noisy rooms.
- Memory across sessions (what it learned about the visitor).

## Asset pipeline (user templates -> generated, checked art)
- `tools/check_face.py` is the deterministic half: files, sizes, transparency,
  alignment. Exit code and `--json` output are meant for automation.
- **Textured mouth style**: `mouth_closed.png` + `mouth_inside.png` with an
  opening mask derived from the lip outline; continuous open/width/round
  from the same smoothed values the procedural mouth uses. Removes the
  six-state flicker and cuts the art a generator must get consistent from
  six images to two. Later: mesh warp of one lip image from keypoints.
- **Vision loop**: user uploads a template (a base image or reference);
  an image model generates the missing parts per `docs/ART_SPEC.md`; a
  vision model plus `tools/check_face.py` judge consistency (same character,
  same style, parts aligned); regenerate until the folder passes.

## Beyond Halloween
- The platform is general: parks, libraries, schools, museums. Faces for
  historical figures and mascots; persona files that carry facts the
  character must stick to; a "docent mode" prompt that answers questions
  about an exhibit and declines off-topic ones gracefully.
- Art from image models via `docs/ART_SPEC.md` (the locked spec).

## Eyes (studied Adafruit Uncanny Eyes, Sept 2026)
- DONE: `talker/textured_eye.py` — saccade-and-hold motion, asymmetric blinks, lid
  tracking, emotion-driven pupil dilation, numpy compositor from iris strip +
  pupil map + lid masks (~1.7 ms/frame for two eyes). Goat and dragon use it
  (`"textured_eye"` in face.json). `EyeMotion.look_at(x, y)` exists for a
  camera or voice direction to drive.
- Apply the motion model to image eyes too (EVE, cat) as whole-eye saccades.
- DONE: `tools/unwrap_eye.py` turns flat eye art into live parts; the
  cat's pupils now move and dilate.
- Pupil size from a light sensor; look-at from a camera.
- **Physical eyes on small round displays** (GC9A01 / OLED modules over SPI
  on the Pi, as in the Uncanny Eyes hardware): the same eye model drives a
  pair of screens in a mask or animatronic head while the face stays
  projected. Roadmap, not current scope; projection is the deployment today.

## Faces
- More faces: EVE ElevenLabs voice id; a second set of mouth art for the cat
  at a finer viseme granularity; minimum hold time for art mouths so fast
  phoneme runs don't flicker.

- **Setup hotspot.** When the Pi has no network, bring up its own Wi-Fi access point with a captive
  page so the control page's Wi-Fi tab can join the venue network without a keyboard.
