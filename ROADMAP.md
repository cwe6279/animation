# Roadmap

Ideas agreed on but not built yet, roughly in priority order.

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

## Beyond Halloween
- The platform is general: parks, libraries, schools, museums. Faces for
  historical figures and mascots; persona files that carry facts the
  character must stick to; a "docent mode" prompt that answers questions
  about an exhibit and declines off-topic ones gracefully.
- Art from image models via `faces/ART_SPEC.md` (the locked spec).

## Eyes (studied Adafruit Uncanny Eyes, Sept 2026)
- **Motion model for every face**: saccade-and-hold (hold 0-3 s, jump in
  72-144 ms with smoothstep), asymmetric blinks (close 36-72 ms, open at half
  speed, next blink 3x duration + 0-4 s), upper eyelid tracking the pupil,
  pupil dilation that never sits still (recursive split noise). Timing rules
  on the offsets we already have; no new art needed.
- **Textured eye type**: sclera + polar iris textures and eyelid masks, all
  numpy lookups (~50k pixels per eye, real time on a Pi 5). Generated default
  textures; faces can supply their own iris/sclera/lid PNGs.
- **Look-at API**: point the eyes at a target (a camera-detected visitor, the
  direction a voice came from); pupil size from a light sensor.
- **Physical eyes on small round displays** (GC9A01 / OLED modules over SPI
  on the Pi, as in the Uncanny Eyes hardware): the same eye model drives a
  pair of screens in a mask or animatronic head while the face stays
  projected. Roadmap, not current scope; projection is the deployment today.

## Faces
- More faces: EVE ElevenLabs voice id; a second set of mouth art for the cat
  at a finer viseme granularity; minimum hold time for art mouths so fast
  phoneme runs don't flicker.
