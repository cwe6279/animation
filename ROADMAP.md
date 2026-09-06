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
- Benchmark Groq (Llama 70B) and OpenAI as the brain *inside the loop* with
  the per-turn `[turn]` line; keep Claude if the character quality gap is
  worth the extra ~0.5 s.
- Pre-open the ElevenLabs websocket for Flash sessions (saves ~100 ms).
- Shorter system prompt variant for the voice loop.

## Raspberry Pi and kiosk deployment
- **Wi-Fi and network setup without a keyboard**: first-boot captive portal or
  a config file on the boot partition (SSID, password, API keys), plus a
  status face state for "no network".
- Cloud speech-to-text by default there (`--stt elevenlabs` or `--stt groq`).
- Face at 30 fps; measure the renderer on the Pi 5.
- Systemd unit + `--fullscreen` autostart; watchdog that restarts on crash.

## Conversation
- Barge-in without headphones: echo cancellation (WebRTC AEC / PipeWire
  echo-cancel module) so the mic ignores the speaker.
- Wake word or push-to-talk button for noisy rooms.
- Per-face `character.md` for longer personas (backstory, catchphrases).
- Memory across sessions (what it learned about the visitor).

## Beyond Halloween
- The platform is general: parks, libraries, schools, museums. Faces for
  historical figures and mascots; persona files that carry facts the
  character must stick to; a "docent mode" prompt that answers questions
  about an exhibit and declines off-topic ones gracefully.
- Art from image models via `faces/ART_SPEC.md` (the locked spec).

## Faces
- More faces: EVE ElevenLabs voice id; a second set of mouth art for the cat
  at a finer viseme granularity; minimum hold time for art mouths so fast
  phoneme runs don't flicker.
