# Talker — Phoneme-Synced Animated Face

A real-time animated talking character driven by streaming TTS and phoneme lip sync:
a mascot at a park gate, a historical figure in a museum, a storyteller in a library,
or a Halloween prop.
Type or pipe in text and the face starts speaking the first sentence while the
rest is still being synthesized — or still being written by an LLM.
Designed for projection on black — faces render as floating features with glow effects.

## Features
- **Streaming speech**: sentence-level pipelining, one persistent audio stream,
  frame-accurate timeline. First sound ~0.5 s after the request with edge-tts.
- **LLM-ready**: `speak_stream()` / `--stdin` accept text as it arrives; pipe a
  streaming Claude reply straight into the face (`tools/claude_stream.py`)
- **Pluggable TTS**: edge-tts (free, default) or ElevenLabs (lower latency, raw PCM)
- Phoneme-accurate lip sync via TTS word timestamps + g2p phoneme mapping
- In-window text box — press Enter, type, Enter again to speak; Ctrl+C to interrupt
- Multiple faces: **pumpkin**, **cat**, **skull**, **ghost**, **eve**, **goat**, **dragon**, and **brain**, a voice-only assistant with no face at all (or create your own)
- Emotion tags — `[angry]`, `[happy]`, `[sad]`, `[surprise]`, `[annoyed]` change eye expressions mid-sentence
- Artist-friendly: drop in PNG art and tweak `face.json`; missing art falls back to procedural glowing shapes
- Debug overlay showing viseme, emotion, FPS, audio queue, and time-to-first-audio

## Project Structure

```
talker/                  <- the runtime, as a package
  app.py                 <- window, text box, CLI for speak.py
  voice_loop.py          <- the round trip: mic -> STT -> brain -> voice + face, wake mode, vision hook
  speech_pipeline.py     <- sentences -> TTS backend -> audio timeline + viseme schedule
  tts_backends.py        <- edge-tts (ffmpeg pipe), ElevenLabs Flash (websocket) and v3 (HTTP stream)
  stt_backends.py        <- faster-whisper, Vosk, ElevenLabs Scribe realtime, OpenAI batch
  vision.py              <- optional camera watcher: bursts -> scene deltas for the brain
  audio_engine.py        <- persistent PortAudio stream with a frame-accurate clock; mic input
  phoneme_scheduler.py   <- pure: tags, sentence splitting, g2p, visemes, ScheduleReader
  face_asset_loader.py   <- face.json manifest, art loading, the face renderer (lids, glow, mouths)
  textured_eye.py        <- eye motion model + per-frame compositor for live textured eyes
  frame_governor.py      <- adaptive frame rate
  env_config.py          <- .env loading
  session_log.py         <- mirrors each session to logs/
  brains/
    claude_chat.py       <- multi-turn streaming Claude (default brain)
    ollama_chat.py       <- local brain via Ollama's native API (thinking off)
    openai_compat_chat.py<- same over OpenAI, for comparison
    system_prompt.md     <- the shared delivery rules and performance-tag guide
tools/
  check_face.py          <- validate a face folder against docs/ART_SPEC.md
  unwrap_eye.py          <- turn flat eye art into live textured-eye parts
  uncanny_eye.py         <- render an eye image from an Adafruit Uncanny Eyes design
  bench_stt.py           <- speech-to-text accuracy + speed across backends
  bench_llm.py           <- brains: speed + adherence to the tag spec
  demo_emotions.py       <- cycles the emotion tags on a face
  claude_stream.py       <- one-shot: stream a Claude reply to stdout for `speak.py --stdin`
faces/                   <- one folder per character: face.json, character.md, art, eye/ parts
  _template/  pumpkin/  cat/  skull/  ghost/  eve/  goat/  dragon/  brain/ (voice only)
docs/
  config-guide.html      <- the tuning guide (every setting, measured choices)
  ART_SPEC.md            <- locked art spec for artists and image models
tests/                   <- pytest, no mic or display needed
voice_loop.py            <- launcher: python voice_loop.py ...
speak.py                 <- launcher: python speak.py --text "..." (was talker.py)
ROADMAP.md  requirements.txt  .env.example
```

## Hardware

Tested with: a Logitech C920 webcam, a Samson Go Mic (USB condenser), a Jabra Speak 510
(USB speakerphone) on a Fedora desktop; a Raspberry Pi 5 is the intended kiosk box.

| Setup | Needs | Notes |
|-------|-------|-------|
| **Desktop, developing** | any USB mic and speaker, optional webcam | Local Whisper is fine here. A speakerphone like the Jabra is handy but it hears itself, so keep half-duplex (the default). |
| **Doorstep or room prop** | projector (or any display), a **separate** directional mic and speaker, optional webcam, internet | Separate mic and speaker are what make `--barge-in` possible: point the mic at the visitors and put the speaker behind or beside it, facing away. Aim for a mic level of 500-5,000 on `--mic-test` when someone talks at visitor distance. |
| **Raspberry Pi 5 kiosk** | Pi 5 with 4 GB or more, USB mic and speaker (or a USB audio interface), HDMI projector, camera (USB or Pi camera), internet | Use `--profile pi`: recognition and everything heavy run in the cloud. Local Whisper takes seconds per turn on a Pi; Vosk is the only workable local option. Pi 4 works at about half the speed. Wi-Fi setup without a keyboard is on the roadmap. |

**Placement.** Camera at the visitors' eye level, wide enough to see a small group and what
they hold up; it only needs to see people, not the projection. Keep the projected face out
of the camera's view or it may describe itself. Mic within about a metre of where people
stand; a USB condenser or a small shotgun mic beats a laptop mic. A cardioid mic (the Go Mic
in its cardioid setting) rejects sound from directly behind it, so put the speaker on the
mic's back side, a metre or more away, facing the visitors past the mic. A foam windscreen or
a small baffle behind the capsule helps outdoors but does not replace that placement. A
non-USB mic needs a class-compliant USB audio interface (XLR: Scarlett Solo or Behringer UM2,
with phantom power for condensers; 3.5 mm plug-in-power mics: an adapter that supplies it). Speaker volume moderate:
loud speakers make the mic hear the character, which defeats barge-in and can trigger
false wake-ups.

**Projection.** Black background, `--fullscreen`, face scaled to the display; any projector
works, brighter helps outdoors. Set `fps` to 30 in the face for a Pi.

**Calibrating a new setup (recommended first step).**
`python voice_loop.py --calibrate --mic-device gomic --output-device jabra` measures the room,
plays the character through the speaker and measures the bleed into the mic, then asks you to
talk from the visitor spot. It tells you whether barge-in is viable, recommends
`--barge-in-boost`, flags mic gain that is too low or too hot, and saves everything to
`calibration.json` (per machine, gitignored). From then on `voice_loop.py` uses those devices
and that boost as defaults whenever the flags are not given.

**Measuring speaker bleed by hand.** `python voice_loop.py --mic-test --mic-device gomic --output-device jabra --play "Testing one two three, can you hear me?"`
plays the character's voice three times through the speaker while printing the mic level; lines
tagged `speaker` are what the mic hears from the speaker, lines tagged `mic` are the room and you.
Rearrange until the speaker number is well below your own speaking level, then set
`--barge-in-boost` so the threshold sits between them.

**Barge-in checklist.** Run with `--barge-in --debug`. A barge-in needs half a second of
continuous speech (`--barge-in-ms`, 700 ms) that is louder than the onset threshold times
`--barge-in-boost` (4) while the character talks, and that does not follow the rhythm of
what the speaker is playing: an echo guard compares the mic's loudness pattern with the
outgoing audio and rejects matches (`[echo]` lines in the log). A USB speakerphone's own mic
(the Jabra's, `--mic-device "jabra speak 510 mono"`) cancels its speaker in hardware and is
worth trying when the mic and speaker cannot be separated. If `[barge-in]` lines still appear while
it speaks and nobody is talking, raise the boost to 4, move or angle the mic, lower the
volume, or drop back to half-duplex.

## Configuration at a glance

Four areas, each with a default that works out of the box and options you switch with a flag
(or a face.json key). Keys go in `.env`.

| Area | Default | Options | Flag | Needs |
|------|---------|---------|------|-------|
| **LISTENING** (speech-to-text) | `whisper` local faster-whisper base.en, ~0.8 s after you stop | `elevenlabs` Scribe realtime (cloud, ~0.5 s, the Pi choice) · `vosk` local, light · `openai` cloud batch | `--stt`, `--silence-ms`, `--mic-device` | Whisper/Vosk: nothing · Scribe: `ELEVENLABS_API_KEY` · OpenAI: `OPENAI_API_KEY` |
| **VISION** (camera) | **off** | on with a camera: 3 frames per burst, every 9 s, described by Claude Haiku 4.5 only when the picture changed | `--camera`, `--no-vision`, `--vision-interval`, `--vision-frames`, `--vision-change`, `--vision-model` | `ANTHROPIC_API_KEY`, `opencv-python-headless` |
| **SPEECH, local** | `piper` en_US-hfc_female-medium, ~20 ms to first audio, no key, offline | `--tts piper --voice en_US-ryan-high` (any Piper voice, downloaded once) · `voices.piper` in face.json · tags drive the eyes only |
| **BRAIN** (LLM) | `claude` Haiku 4.5, ~0.7 s to first token | `--model claude-opus-5` best writing, ~2 s more per reply · `--thinking` for deeper answers · `ollama` local, no key · `openai` for comparison | `--llm`, `--model`, `--effort`, `--thinking`, `--character` | `ANTHROPIC_API_KEY` (workspace-scoped, prepaid) · `OPENAI_API_KEY` |
| **SPEECH** (text-to-speech) | `elevenlabs` v3 when the key is set (performs `[tags]`, ~1 s to first audio), else `edge` free | `--tts-model flash` (~0.25 s, tags stripped) · `fish` Fish Audio · `edge` free | `--tts`, `--tts-model`, `--voice`, `--voice-speed`, `--output-device` | ElevenLabs: `ELEVENLABS_API_KEY` · Fish: `FISH_AUDIO_API_KEY` · edge: nothing |

Per face, `face.json` can fix the voice (`voices`), the ElevenLabs model (`tts_model`),
the wake words and the personality lives in `character.md`. `--profile pi` picks the
cloud choices for a Raspberry Pi in one flag. Measured numbers for every option are in
[Choosing backends](#choosing-backends-measured-results).

## How it fits together (latency-first)

```
 text / LLM tokens ──> SentenceSplitter ──> TTS backend (one session per utterance)
                                              │ audio PCM chunks     │ word timestamps
                                              ▼                      ▼
                                       AudioEngine.enqueue_pcm  word_to_viseme_events
                                       (returns timeline time)  placed at session_start + t
                                              │                      │
                                              ▼                      ▼
                                       PortAudio callback     ScheduleReader (append)
                                       frames_out / rate ───────────> renderer asks
                                       = timeline clock               current_viseme(t)
```

- The output stream is opened once and never stops; silence is emitted when idle,
  so the clock is continuous and utterances queue back to back with no gaps.
- Each utterance's first PCM chunk tells the pipeline where on the timeline it
  starts; word timings from the backend are offset by that, so lip sync does not
  depend on wall-clock guesses.
- Sentences are sent to TTS as soon as the splitter sees terminal punctuation,
  so an LLM's first sentence is spoken while it is still writing the rest.

## Setup

### System dependencies
```bash
# Mac
brew install ffmpeg portaudio

# Ubuntu/Debian/Fedora
sudo apt install ffmpeg portaudio19-dev      # Fedora: sudo dnf install ffmpeg portaudio-devel

# Windows
# ffmpeg: pip install imageio-ffmpeg (bundled, no PATH setup needed)
# PyAudio: pip install pyaudio
```

### Python packages
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- `pygame-ce` — rendering (drop-in pygame fork with wheels for Python 3.13/3.14)
- `pyaudio` — audio output/input
- `numpy` — amplitude analysis, WAV conversion
- `edge-tts` — default TTS with word timestamps (needs internet); MP3 decoded by ffmpeg
- `g2p-en` — grapheme-to-phoneme (optional, falls back to built-in regex)

Run the tests any time (no sound device or display needed):
```bash
pytest tests
```

## Usage

```bash
# Speak text with a face
python speak.py --face pumpkin --text "Happy Halloween!"
python speak.py --face cat --text "Meow, I am a spooky cat"
python speak.py --face skull --text "I am a talking skull"
python speak.py --face eve --text "[happy]Wall-E? [surprise]Directive!"

# Just open the window and type (Enter focuses the text box)
python speak.py --face cat

# Debug overlay (viseme, emotion, FPS, audio queue, time-to-first-audio)
python speak.py --face pumpkin --text "Hello" --debug

# Emotion tags inline / default emotion for the whole utterance
python speak.py --face cat --text "[angry]I am very angry! [sad]But also sad."
python speak.py --face cat --text "Hello world" --emotion surprise

# Stream text in as it is produced (LLM output, another program, a script)
python tools/claude_stream.py "Tell me a two-sentence ghost story" | python speak.py --face skull --stdin

# ElevenLabs instead of edge-tts (set ELEVENLABS_API_KEY; --voice is a voice id)
python speak.py --tts elevenlabs --voice 21m00Tcm4TlvDq8ikWAM --text "Hello there"

# Lip-sync tuning: shift the face relative to audio (seconds), mouth lead time
python speak.py --text "Testing sync" --sync-offset -0.03 --lead 0.05

# Projection: fullscreen, face scaled to the display, no overlay or cursor (F toggles)
python speak.py --face skull --fullscreen --text "..."
python voice_loop.py --face eve --fullscreen

# Auto-exit after speech (for scripts) / emotion demo
python speak.py --face cat --text "Quick test" --auto-exit
python tools/demo_emotions.py

# Play an existing WAV (amplitude mode) / live microphone
python speak.py --face pumpkin --file my_audio.wav
python speak.py --face pumpkin --mic

# Custom face from any directory; unknown names get a procedural default face
python speak.py --face-dir path/to/my_face --text "Hello"

# No sound device (CI, headless): render only
python speak.py --no-audio --auto-exit --text "Hello"
```

**Keyboard shortcuts in window:**
- `Enter` or `T` — focus the text box; type; `Enter` speaks it (`Up/Down` recall history)
- `Ctrl+C` — interrupt: stop speaking and flush queued audio
- `Esc` — unfocus the text box, or quit
- `D` — toggle debug overlay, `H` — toggle hints/text box, `F` — toggle fullscreen
- `Space` — waiting mode on/off: stops talking, ignores the mic, wake words and typed text, dims the face; for a call or a meeting. Nothing automatic leaves it; the control page has the same switch

## Talking to it (the round trip)

### First run: `--setup`

On a new machine, one command does the whole thing in order and tells you what is wrong
before anything else can go wrong:

```bash
python voice_loop.py --setup
```

It checks the packages and ffmpeg, validates each API key **against the service** rather than
just noting that it is set, then walks you through picking and testing the speaker, the
microphone and the camera, offers to measure the room, and finishes with the exact command to
run. The devices it picks are saved to `calibration.json`, so the flags are optional afterwards.

The key check is the part worth having. Without it a missing or uncredited Anthropic key is
not noticed until the character first tries to answer, and all you hear is "Sorry, I could not
think of an answer", which tells you nothing.

### API keys

Copy `.env.example` to `.env` and fill in your own keys. `.env` is gitignored;
never commit it or paste keys into code.

| Key | Used for | Where to get it |
|-----|----------|-----------------|
| `ANTHROPIC_API_KEY` | Claude replies (`voice_loop.py`, `claude_stream.py`) | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys). Create the key **inside a workspace**; the API is prepaid, so add credits under Plans & Billing first. |
| `ELEVENLABS_API_KEY` | ElevenLabs voice (`--tts elevenlabs`), optional | [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys) |
| `ANTHROPIC_WORKSPACE_ID` | Only if your Anthropic key is organization-wide | Workspace settings page in the Console |

edge-tts, Vosk and faster-whisper need no keys. If you use the Anthropic CLI, `ant auth login` works instead of the key.

### Run it

```bash
python voice_loop.py --face cat --mic-device gomic --output-device jabra --debug
```

Defaults: ElevenLabs v3 voice when `ELEVENLABS_API_KEY` is set (else edge-tts),
faster-whisper for speech-to-text, and the face's own voice id and persona from
its face.json. Every flag below overrides those.

Speak; when you pause, the text is sent to Claude and the reply is spoken as it
streams in. Every turn prints where the time went (STT final, first LLM token,
first audio). Options:

- `--mic-device NAME` / `--output-device NAME` — pick devices by a name fragment (`gomic`, `jabra`);
  `--list-devices` shows them. Numbers work too but shift when devices are plugged in.
- `--mic-test` — print levels and transcripts only, to check a mic before going live
- `--tts elevenlabs --voice <id>` — ElevenLabs voice (needs `ELEVENLABS_API_KEY`)
- `--tts fish --voice <reference id>` — Fish Audio voices (`FISH_AUDIO_API_KEY`; API credit is separate from
  site credit). No word timestamps from the service, so words are fitted to each sentence's measured length;
  a sentence of latency, lip sync within a syllable. Set a face default with `"voices": {"fish": "<id>"}`.
- `--tts-model flash` — the fast ElevenLabs model (~0.25 s to first audio versus ~1 s for the default v3; tags stripped)
- `--fixed-fps` — pin the frame rate. By default a frame governor steps the face down to 45/30/20/15 fps
  when frames run over budget (a busy Pi) and back up once there is headroom; lip sync is unaffected
  because timing comes from the audio clock. The debug overlay shows the current target.
- `--stt elevenlabs` — cloud speech-to-text (ElevenLabs Scribe realtime). Server-side endpointing,
  committed transcript ~0.5 s after you stop, zero local CPU: the choice for a Raspberry Pi.
- `--stt vosk` — light local recognizer (`--vosk-model lgraph|large` for better accuracy)
- `--stt whisper` — the default: local faster-whisper (accurate; ~300 ms per turn on a desktop CPU, seconds on a Pi)
- `--silence-ms 400` — how long a pause ends your turn (default 600); lower is snappier, cuts more
- `--thinking` — turn on Claude's reasoning pass (off by default; adds about a second to the first token)
- `--model claude-haiku-4-5` — fastest replies, a little less wit
- `--barge-in` — keep listening while it speaks and interrupt when you talk (use headphones,
  otherwise the speakers get transcribed). Default is half-duplex: mic ignored during playback.
- `--text-only` — type in the window instead of using a mic; same Claude round trip
- `--effort medium` — better answers, slower first token. `--model` to change the model.
- `--tts piper` — the local voice: no key, no network, lip sync from the model's own phoneme timings. `--voice` picks any [Piper voice](https://github.com/OHF-Voice/piper1-gpl/blob/main/VOICES.md) (or `voices.piper` in face.json, or `PIPER_VOICE`); it downloads once. Measured against the other engines in *Local voices* below
- `--llm ollama --model <name>` — a local brain through [Ollama](https://ollama.com): no key, no cloud, thinking
  forced off so the first token is quick (well under a second on a desktop GPU, after a one-time load).
  `--list-models` shows what the server has; `--ollama-host` or `OLLAMA_HOST` points at a remote server.
- `--llm openai` — OpenAI as the brain, for side-by-side comparison; `--model` picks the model
- `--stt openai` — OpenAI batch speech-to-text, for comparison

The pieces are independent: `VoiceLoop` (voice_loop.py) only needs an STT object,
a function that returns an iterator of reply text, and something with
`speak_stream` / `interrupt` / `is_busy`. Swap any of them.

### Projection

`--fullscreen` (or F in the window) is exclusive fullscreen: the face scaled to the display by
the GPU, black bars if the aspect differs, no cursor or overlay. `--borderless` projects as a
frameless window the size of the desktop instead: no display mode switch, so no compositor
flicker or frame flashes at start-up; the face is scaled in software, about 1 ms a frame on a
desktop. Use `--borderless` when fullscreen shows artifacts. For pixel-exact edges either way,
set `canvas_w`/`canvas_h` in face.json to the projector's resolution and scale the coordinates.

### The control page (on by default, port 8020, this machine only)

Every run serves a small page at `http://localhost:8020` (`--web-port`, `--no-web`; the next free port if that one is
taken), standard library only, off the audio and render paths. It binds to this machine only.
`--web-host 0.0.0.0` opens it to the network so you can use it from a phone, which is what a kiosk
wants and what `docs/talker.service` does; be deliberate about it, because the page has no password
and serves a camera snapshot and a Wi-Fi join endpoint. What it gives you:

- **Status**: face, ears, brain, voice, the live mic level, speaking / thinking / dormant, the last
  `[turn]` timing line, what was heard and said, vision counts and the latest scene note.
- **Tune**: the live settings with their help text and the flag that sets each at startup: the
  silence gate, barge-in and its threshold and boost, the echo guard, idle timeout, engaged, the
  debug overlay, the vision interval and change filter. Changes apply at once.
- **Test setup**: mic level meter, speak a line through the speaker (tags work), send a line as a
  visitor, interrupt, a camera snapshot, and the calibration routine (room, speaker bleed, a
  person) with its verdict, saved to `calibration.json` like `--calibrate`.
- **Reference**: every `--flag` with its help, default and the value in use this run, and the
  face.json fields of this face with what they do.
- **Wi-Fi**: the box's network devices, a scan, and join a network through NetworkManager (as on
  Raspberry Pi OS), for kiosks with no keyboard. It needs the box to be reachable first, over
  Ethernet or a known Wi-Fi; a captive setup hotspot is on the roadmap.

### Raspberry Pi

**Getting it on Wi-Fi the first time.** Use Raspberry Pi Imager and open its advanced options
(the gear) before writing the card: set the Wi-Fi name, password and country, and set a hostname.
The Pi joins the network on first boot with no keyboard or monitor, and the control page is then at
`http://<hostname>.local:8020` from any phone on the same network, which is what
`docs/talker.service` exposes. Changing networks later is the control page's Wi-Fi tab. If the card
was flashed without Wi-Fi details, plug in Ethernet once, or a keyboard and monitor, to get there;
a self-hosted setup hotspot for that case is on the roadmap.


The face and audio plumbing are light; the recognizer is the only stage a Pi
cannot run fast. One flag picks the right set:

```bash
python voice_loop.py --face cat --profile pi --mic-device gomic --output-device jabra
```

`--profile pi` = ElevenLabs Scribe realtime for speech-to-text, fullscreen,
30 fps. Any flag you pass explicitly still wins,
e.g. `--model claude-haiku-4-5` for a faster brain. Tested on a Pi 5 with
4 GB or more; a Pi 4 works but everything local is about twice as slow.
Cloud stages cost the same on a Pi as on a desktop.

**Start at power-up.** `docs/talker.service` is a systemd unit that launches the character
with the control page when the Pi boots; copy it to `/etc/systemd/system/`, edit the user, path
and flags, then `sudo systemctl enable --now talker`. `journalctl -u talker -f` shows the log.

### Wake mode (kiosks: wait to be called by name)

```bash
python voice_loop.py --face eve --wake                    # wake words from face.json, else the face name
python voice_loop.py --face eve --wake-word "eve, hey eve, hello eve"
```

With `--wake` the character starts engaged, then after `--idle-timeout` seconds
of silence (or a goodbye) goes dormant: it listens and, if vision is on, keeps
watching, but answers nothing until a wake word is heard. `--start-dormant`
makes it wait to be called from the start (the Pi profile does this). List
likely mis-hearings in `wake_words` ("marsha", "martha", "marcia"). Whatever follows the name in the same sentence is
answered at once; the name alone gets a reply to being called. It goes
dormant again after `--idle-timeout` seconds of silence after the character last spoke (60) or when the brain
ends the conversation: the prompt asks it to finish a farewell with the marker
`[end]`, which the loop strips before the voice. `wake_words` in face.json
sets the defaults per character; `--profile pi` turns wake mode on. **Sleep words** put it to
sleep at once and cut it off mid-sentence: "stop", "wait", "hold on", "hang on", "pause",
"quiet", "go to sleep", "enough", "goodbye" and a few more, when said on their own (up to four
words); `--sleep-word "..."` or `sleep_words` in face.json override the list. With `--barge-in`
a sleep word works while the character is still talking.

Sounds are not words: Whisper runs behind a voice-activity filter and a confidence cut-off, and
clips under 0.35 s are dropped, so coughs, chair scrapes and music no longer become sentences.

### Vision (optional, off unless `--camera` is given)

```bash
python voice_loop.py --list-cameras
python voice_loop.py --face eve --camera c920 --debug
```

Every 9 s (`--vision-interval`) the watcher takes 3 frames (`--vision-frames`)
about 0.3 s apart, downsizes them, and asks a fast vision model
(`--vision-model`, default Claude Haiku 4.5, roughly a sixth of a cent per
burst) for a few lines of notes: how many people, rough ages, what they are
doing or holding, mood. A burst is also taken the moment a visitor starts
talking, so the note is fresh by the time the transcript lands; a visual
question ("what's this", "can you see", "how many") waits up to 2.5 s for it.
When a burst changes the scene, the delta enters the conversation as an inner
observation, "(You notice: ...)", ahead of the next thing the visitor says, and
the prompt tells the character it is its own eyesight, not text to recite:
react in your own words only if it matters, never repeat the observation.
Quiet turns add nothing, and the visitor's words are never altered. A burst is only sent to the model when
the scene has changed (a tiny thumbnail is compared with the last described
one; `--vision-change`, default 0.06, a still room scores ~0.01), with a
forced refresh every 90 s, so an empty room costs nothing. Frames are
discarded as soon as they are described; only if the model flags an emergency (someone hurt or in
distress, fire, a clear hazard) are that burst and its note saved under
`emergencies/<timestamp>/` (gitignored) and the brain told to stay calm and
call an adult. `--no-vision` forces it off. Needs `opencv-python-headless`.

### Speech-to-text options, measured on a desktop

| `--stt` | Runs | Transcript ready after you stop | Accuracy | Raspberry Pi |
|---------|------|-------------------------------|----------|--------------|
| `whisper` (default) | local CPU | ~0.7 s (0.4 s silence + 0.3 s transcribe) | high | 2–5 s per turn |
| `vosk` | local CPU | ~0.5 s, live partials | fair (small) / good (large) | fine |
| `elevenlabs` | cloud | ~0.5 s, live partials | high | fine (no CPU) |
| `openai` | cloud batch | silence wait + ~0.5–1 s | high | fine (no CPU) |

Every turn prints one `[turn]` line: time from when you stopped talking to the
transcript, to the first LLM token, and to the first audio. Use it to compare
backends in the real loop rather than in isolation.

### Choosing backends: measured results

Measured 2026-09-06 on a recent many-core AMD desktop, CPU only, on a fast
connection. Rerun on your own hardware and voice
with the two benchmark utilities:

```bash
python tools/bench_stt.py --synth                       # recognizers on synthesized clean + noisy clips
python voice_loop.py --mic-test --record recordings/ --mic-device gomic   # record your own clips
python tools/bench_stt.py recordings/                   # recognizers on your clips (correct transcripts.txt first)
python tools/bench_llm.py                               # brains: speed + how well they follow the tag spec
```

A Raspberry Pi will be far slower on anything marked *local*; cloud rows are
unchanged there.

**Speech-to-text** — 12 clips (6 clean, 6 with noise at 10 dB SNR), word
error rate and seconds per clip. Synthesized voices are cleaner than a real
mic, so expect higher error rates in a room.

| `--stt` | backend | WER | clean | noisy | s/clip | runs |
|---------|---------|----:|------:|------:|-------:|------|
| `whisper` (default) | faster-whisper base.en | 2.1% | 1.4% | 2.8% | 0.19 | local |
| `whisper --whisper-model small.en` | faster-whisper small.en | 2.7% | 1.4% | 4.2% | 0.48 | local |
| `vosk` | vosk small | 4.1% | 1.4% | 7.0% | 0.19 | local, live partials |
| `openai` | whisper-1 | 1.4% | 1.4% | 1.4% | 1.39 | cloud |
| `openai --model gpt-4o-mini-transcribe` | gpt-4o-mini-transcribe | **0.0%** | 0.0% | 0.0% | 0.77 | cloud |
| `elevenlabs` | Scribe (batch v1 measured; the loop uses realtime) | 1.4% | 1.4% | 1.4% | 0.51 | cloud, live partials, server VAD |

Recommendation: local Whisper on a desktop (free, ~0.8 s after you stop);
ElevenLabs realtime on a Pi (~0.5 s after you stop, no local CPU). Vosk only
if you must stay offline on weak hardware.

**Brains** — same cat persona, 8 prompts each. *known* = share of tags in the
allowed vocabulary; *leading* = tags placed before words; *tags/sent* = tags
per sentence (the prompt asks for most sentences to have none); *md* =
replies with markdown or emoji.

| `--llm` / `--model` | first token | known | leading | tags/sent | md | words | notes |
|---------------------|------------:|------:|--------:|----------:|---:|------:|-------|
| `claude` claude-opus-5 (default, thinking off) | 940 ms | 100% | 100% | 0.69 | 0% | 25 | best writing: specific, witty, in character |
| `claude --model claude-haiku-4-5` | 650 ms | 100% | 100% | 0.75 | 0% | 26 | good; longer, occasional *asterisk* emphasis |
| `openai` gpt-4o-mini | 500 ms | 100% | 100% | 0.54 | 12% | 22 | compliant; chirpy, many exclamation marks |

`--thinking` adds ~1 s to first token; `--model claude-haiku-4-5` is the
faster lever. Every model tags more than the
prompt asks; tighten rule 3 in `talker/brains/system_prompt.md` if it feels
busy.

**Voices** — time to first audio through the whole pipeline, ElevenLabs on a
warm connection.

| `--tts` / `--tts-model` | first audio | performs `[tags]` | timing data |
|-------------------------|------------:|-------------------|-------------|
| `elevenlabs` eleven_v3 (default) | ~0.8–0.9 s | yes | per character |
| `elevenlabs --tts-model eleven_flash_v2_5` | ~0.25 s | no (stripped) | per character |
| `edge` (free) | ~0.2–0.5 s | no (stripped) | per word |
| `piper` (local, offline) | ~25 ms after the sentence | no (stripped) | per phoneme, from the model |

### Going local: where the milliseconds go

Every stage can run on the box or in the cloud. What a listener waits is the sum of the
stages on the critical path, so pick per stage. Measured on the desktop above (September 2026);
a Pi 5 runs the local rows 3-4x slower, the cloud rows the same.

| stage | cloud option | local option | what it costs on the critical path |
|---|---|---|---|
| listening | ElevenLabs Scribe realtime, ~0.5 s after you stop | faster-whisper base.en, ~0.2 s after the silence gate (~0.8 s total) | the `--silence-ms` gate (600) is the largest fixed cost of the turn; 400 is snappier, 300 cuts pauses |
| brain | Claude Haiku 4.5, ~650 ms to first token; Opus ~950 | Ollama: `qwen3:8b` ~220 ms on a desktop GPU, a 27B-class model 400-450 ms; hybrid-attention families (Qwen 3.5/3.8, Gemma) cannot reuse the prompt cache, so every turn re-reads the prompt (~700 ms, with multi-second outliers) | the first *sentence* gates the voice, so a brain that opens short wins; thinking is always off (adds ~1 s) |
| voice | ElevenLabs flash ~250-400 ms, v3 ~1 s (performs [tags]) | Piper ~25 ms; Kokoro ~0.8 s for the first sentence (better voice) | Piper is the only voice that adds nothing you can hear |
| vision (optional) | Claude Haiku, ~2.2 s per look, off the critical path | `gemma4:31b` via Ollama, ~10 s per look and it shares the GPU with the brain | keep it in the cloud; it runs between turns, never in front of a reply |

**Recommended stacks**

- **Desktop, fastest with a good voice:** Whisper + Claude Haiku + ElevenLabs flash, or `--tts piper` to
  drop the voice's network round trip. About 1.0-1.1 s from transcript to first audio.
- **Desktop, everything local, no keys:** Whisper + `--llm ollama --model qwen3:8b` + `--tts piper`.
  About 0.75 s from transcript to first audio, the fastest pair measured, at the price of a plainer
  voice and a smaller brain. Choose a standard-attention model so the prompt cache holds.
- **Desktop, best performance:** Claude Haiku (or Opus for the writing) + ElevenLabs v3. The tags are
  performed, the eyes and the voice agree, and you pay ~1.7 s to first audio.
- **Pi 5 kiosk:** `--profile pi` (Scribe realtime + Claude Haiku) with `--tts piper` for an instant,
  offline voice, or ElevenLabs flash when the voice matters more. A local brain on the Pi is
  untested and would be several seconds per reply; leave the brain in the cloud.

Measured with `tools/bench_e2e.py` (brain + voice pairs, below), `tools/bench_tts_local.py`
(nine offline voices, below), `tools/bench_stt.py` and `tools/bench_llm.py` (above).

### Local voices (offline TTS), measured

`tools/bench_tts_local.py` runs nine open-source engines on the same sentences, CPU only, 4 threads
each (a Pi 5 has 4 cores; expect roughly 3-4x these desktop times there), and saves the WAVs to
`logs/tts_bench/` so you can listen. It needs its own Python 3.11 environment because most of these
do not support 3.14: `uv venv --python 3.11 .bench-venv`, then install `piper-tts kokoro-onnx
onnxruntime soundfile coqui-tts[codec] ChatTTS` and the MeloTTS git package (both gitignored).
Desktop results, many-core AMD desktop, September 2026 (`ttfa` = time to first audio, `rtf` = synthesis
time / audio time, lower is better):

| engine | load s | RAM MB | ttfa short | ttfa medium | rtf medium | rtf long | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **piper** | 0.9 | 342 | 19 ms | 25 ms | 0.02 | 0.01 | fastest neural voice by far; clear, a little flat; streams per sentence |
| **kokoro** | 0.4 | 776 | 225 ms | 767 ms | 0.11 | 0.12 | best quality that still fits a Pi; streams per sentence |
| melo | 6.7 | 2468 | 305 ms | 962 ms | 0.13 | 0.19 | Kokoro-class cost, older sound, 2.5 GB RAM |
| vits | 16.5 | 1246 | 208 ms | 896 ms | 0.09 | 0.09 | one LJSpeech voice, decent |
| tacotron2 | 8.3 | 1261 | 622 ms | 4519 ms | 0.26 | 0.25 | 2018-era; babbles on longer text |
| xtts | 150 | 4370 | 3234 ms | 12501 ms | 1.33 | 1.38 | voice cloning; slower than real time on CPU |
| chattts | 2.6 | 1858 | 3071 ms | 14309 ms | 1.84 | 1.91 | very natural; twice slower than real time on CPU |
| flite | 0.0 | 41 | 23 ms | 56 ms | 0.01 | 0.01 | Festival family; instant, robotic |
| espeak | 0.0 | 42 | 7 ms | 12 ms | 0.00 | 0.00 | instant, robotic, 100+ languages |

Reading it for the Pi: anything with an rtf above about 0.3 here will not keep up there, which rules
out XTTS, ChatTTS and Tacotron2. Piper and Kokoro are the two real candidates: Piper when the first
word must come instantly, Kokoro when the voice matters more and 0.7-1 s before the first sentence
is acceptable. None of them performs the `[tags]` ElevenLabs v3 does; the tags still drive the eyes.

### Question in, voice out: local vs cloud, measured

`tools/bench_e2e.py` runs the same questions through real brain + voice pairs and times what a
listener waits from the moment the transcript reaches the brain (speech-to-text is the same local
Whisper in every pair, so it is left out). Desktop, September 2026, medians over four questions:

| pair | brain | voice | first token | first audio | reply synthesized |
|---|---|---|---:|---:|---:|
| cloud | claude-haiku-4-5 | ElevenLabs flash | 682 ms | 1098 ms | 1754 ms |
| local | Ollama, a 27B-class model on a desktop GPU | Piper | 426 ms | 763 ms | 1002 ms |
| cloud brain, local voice | claude-haiku-4-5 | Piper | 637 ms | 999 ms | 1434 ms |
| local brain, cloud voice | Ollama | ElevenLabs flash | 442 ms | 894 ms | 1164 ms |

What it says: the wait to first audio is mostly the brain writing its first sentence; the voice
adds about 25 ms with Piper and 300-400 ms with ElevenLabs flash. On a Pi the brain stays in the
cloud (a model that size needs a desktop GPU), so the realistic Pi pairs are the two Claude rows: Piper
takes roughly 100-300 ms off first audio and removes the network from the voice entirely, at the
price of a plainer voice and no performed tags.

### Using it from Python

```python
# app is a TalkerApp (see talker.main for construction); run it on the main thread
app.speak("[happy]Hello!")           # whole string, returns immediately
app.speak_stream(token_iterator)     # any iterable of text chunks, e.g. an LLM stream
app.interrupt()                      # barge-in
```

`SpeechPipeline` (talker/speech_pipeline.py) is independent of pygame: give it an
audio engine, a `ScheduleReader`, and a backend, and read
`current_viseme(t)` from any renderer.

---

## Performance Tags

Text can carry `[bracketed]` performance tags in the ElevenLabs v3 style: emotions,
reactions, tone and character cues (`[excited]`, `[sigh]`, `[light chuckle]`,
`[British accent]`). Two things happen to them:

- **Voice.** With `--tts elevenlabs --tts-model eleven_v3` the voice performs them
  (a real sigh, a laugh, a whisper). Every other backend gets them stripped, so a
  plain voice never reads "sigh" aloud.
- **Face.** Tags with an emotional meaning are mapped onto the six eye expressions
  below (`[excited]` → happy, `[frustrated]` → angry, `[gasps]` → surprise, `[sigh]` →
  sad, ...). Tags like `[pauses]` or an accent leave the eyes alone. The map is
  `TAG_TO_EMOTION` in talker/phoneme_scheduler.py.

v3 costs about half a second more to first audio than Flash v2.5, so it is a
trade: expressive delivery versus snappiness.

### Eye expressions

The six eye states (squish, tilt, blink rate) that tags map onto:

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
python speak.py --face cat --text "Everything is fine." --emotion angry
```

### Emotion Demo

`tools/demo_emotions.py` cycles through all emotions with example sentences:

```bash
python tools/demo_emotions.py
```

Each demo auto-exits after speech finishes. Press ESC to skip, Ctrl+C to stop all.

### LLM Integration

The [talker/brains/](talker/brains/) folder contains a ready-to-use system prompt (`system_prompt.md`) that teaches an LLM how to use emotion tags correctly, plus `claude_stream.py`, which streams a Claude reply to stdout so it can be piped into `speak.py --stdin`. The face starts speaking the first sentence while the model is still generating; `--effort low` keeps replies short and quick for voice.

For a full voice loop (mic → speech-to-text → LLM → face) call `app.speak_stream(...)` with the LLM's token iterator and `app.interrupt()` when the user starts talking again.

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
python speak.py --face my_face --text "Testing my new face" --debug
```

Tweak `face.json` without re-exporting PNGs:
- **`cx`, `cy`** — reposition eyes or nose (pixels from top-left)
- **`scale`** — resize eyes, nose, or mouth art (0.5 = half size, 2.0 = double)
- **`opacity`** — transparency per feature (0.0 = invisible, 1.0 = fully opaque)
- **`anchor_cx`, `anchor_cy`** — mouth position (also affects procedural fallback)
- **`offset_x`, `offset_y`** — shift mouth art without changing the anchor
- **`voices`** — default TTS voice per backend, e.g. `{"elevenlabs": "<voice id>", "edge": "en-US-AriaNeural"}`; `--voice` overrides
- **`{placeholders}` in `character.md`** — filled in once when the character starts, so she knows the date, time and anything else you want to hand her: `{date}` `{iso_date}` `{time}` `{datetime}` `{weekday}` `{year}` `{timezone}` `{utc_offset}`. Add your own per face with `"facts": {"venue": "the north gate"}` in face.json, which gives `{venue}`, or in code with `launch_facts.register("weather", fn)`. Unknown placeholders and `{{action blocks}}` are left alone. Filled at launch rather than per turn so the brain's prompt cache still holds; a kiosk left running past midnight keeps the date it started with
- **`tts`** — this face's own voice backend when `--tts` is not given: `piper` (local, no key), `elevenlabs`, `edge`, `fish`. Clara uses `piper`
- **`tts_model`** — ElevenLabs model for this face: `"v3"` (performs tags, ~1 s to first audio) or `"flash"` (~0.25 s, tags stripped); `--tts-model` overrides
- **`character.md`** (a file next to face.json) — the personality: who the character is, traits, tone, boundaries. Concatenated with the shared `talker/brains/system_prompt.md` at runtime, so the shared file holds delivery rules and the face folder holds only personality. `--character` on the command line overrides it for one run
- **`gaze`** / **`blink_interval`** / **`blink_speed`** / **`eye_speech_pulse`** — idle glances, blink timing, and eye pulse while speaking (image and procedural eyes)
- **`draw_eyes: false`** — no eyes at all; with mouth `opacity: 0` the character is voice only (the `brain` face)
- **Ambient sounds when idle** — files in `faces/<name>/sounds/idle/` (or, without that subfolder, the `sounds/` effects themselves) play one at a time at random gaps while nothing is happening: not speaking, not thinking, nobody talking for `quiet_for` seconds. `"idle_sounds": {"interval": [25, 70], "quiet_for": 10}` in face.json sets the gap range and the quiet time; both are live on the control page's Tune tab. The folder is the switch: no files, no ambience
- **Actions next to speech** — the brain can write `{{move nod}}`, `{{sfx creak}}` or `{{tool clock}}` inside its reply. Blocks are stripped in the same pass as the `[emotion]` tags, so nothing extra sits between the brain and ElevenLabs, and each one fires at the moment the words before it are spoken (barge-in cancels the rest). `sounds/` next to face.json holds the sound effects (wav/ogg/mp3, the file name is the sound name). A sound waits until she has finished the sentence rather than muddying her voice, and is dropped if she has started another reply by then; `"sfx_over_speech": true` in face.json plays it at its word instead; `"body": {"moves": ["nod", "shake"]}` lists the movements (logged for now; servo drivers plug into `talker/body.py`); tools are Python callables registered in `voice_loop.py`, and what they return reaches the brain as an observation on the next turn. The brain is only told about what the face actually has, so a face with no sounds and no body sees none of this
- **`eye_lids`** — for image eyes without pupils (EVE): emotions become lid cuts, a happy crescent, an angry slant, a sad droop, and blinks close the lids instead of squashing the eye
- **Making your own eye art live**: `python tools/unwrap_eye.py faces/<name>/eye_right.png faces/<name>/eye`
  finds the pupil, unwraps the iris into a polar strip, cuts a pupil map in the pupil's real shape,
  derives lid masks from the outline and keeps the reflections as a fixed overlay; it prints the
  `textured_eye` block to paste into face.json. `--pupil-map` borrows another design's map if the
  detected one is off. The cat's original art is converted in `faces/cat/eye_unwrapped/`; she currently uses the dragon design recoloured green.
- **`textured_eye`** — live eyes composed from parts in a folder (`{"dir": "eye", "size": 224, "gaze_radius": 0.35, "pupil": [min, base, max], "lid_tracking": 0.35}`): the pupil and iris move inside a fixed outline on saccades, the pupil dilates with emotion, the upper lid follows the gaze, blinks close fast and open slow. Parts: `iris.png` (polar strip), `pupilMap.png`, `lid-upper.png`, `lid-lower.png`, optional `sclera.png` and `highlight.png`. Any Adafruit Uncanny Eyes design folder works as-is
- **`glow_color`** — halo color around procedural shapes
- **`glow_style`** — `halo` (default) spills soft light outward around each shape; `inner` keeps the cut edges crisp and lights them from inside, a hot core fading to the shape colour like a candle behind a carved pumpkin. `core_color` sets the hot spot (default: the shape colour pushed toward white), `rim_color` draws a thin cut-edge line, `light_offset` moves the hot spot down (0.15). `cut_depth: [8, 6]` insets the lit shape and shows the shell's inner wall along the other side in `wall_color` (pale yellow) for a 3D carved look; `[0, 0]` turns it off. The `pumpkin` face uses `inner`
- **`mouth.style`** — `toothed` (zigzag), `rounded` (oval), or `grin`: a carved smile with the corners turned up that stays as a curved band at rest and opens from the middle while speaking, with up to four goofy square teeth (`n_teeth`). Procedural shapes scale with `canvas_w`, so a face can be set to the projector's resolution (e.g. 1080) for pixel-exact edges in fullscreen
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
Text input (with [emotion] tags)
  |
SentenceSplitter  -> sentences as soon as terminal punctuation is seen
  |
TTS backend (streaming)
  -> PCM audio chunks         -> AudioEngine queue (persistent PortAudio stream)
  -> word timestamps          e.g. ("Happy", 0.24s, 0.62s), ("Halloween", 0.68s, 1.31s)
  |
g2p-en (or built-in regex fallback)
  -> ARPAbet phonemes per word: "Happy" -> [HH, AE, P, IY]
  |
Phoneme -> Viseme map (12 shapes)  -> [AH, AA, PP, EE]
  |
Spread across the word (vowels get more time), offset by where the utterance
landed on the audio timeline, minus a small lead so easing peaks on the beat
  -> ScheduleReader.append([...VisemeEvent], [...EmotionEvent])
  |
Each frame: t = AudioEngine.timeline_time()  (frames played / rate - device latency + --sync-offset)
            renderer.update(current_viseme(t), current_emotion(t))
```

### TTS backends

| Backend | Cost | Time to first audio | Timing data | Notes |
|---------|------|---------------------|-------------|-------|
| `edge` (default) | free | ~0.5 s (server dependent) | word boundaries | MP3 only; decoded through an ffmpeg pipe as it streams |
| `elevenlabs` | paid | ~0.5–0.8 s measured (incl. websocket connect) | character alignment | raw PCM websocket, no decode step |

ElevenLabs' websocket returns PCM and per-character timings, so it removes the
ffmpeg decode and gives tighter lip sync. Edge stays the free fallback. Adding another
service means one class in `talker/tts_backends.py` that yields `AudioChunk` and
`WordBoundary` events (see `TTSBackend`).

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

## Credits

Talker stands on other people's work.

- **[Adafruit Uncanny Eyes](https://github.com/adafruit/uncanny_eyes)** — the live eye design: a
  polar iris strip, a distance field that shapes the pupil, and grey lid masks, with the motion
  model (saccade timing, dilation, blinks that close fast and open slow) taken from
  `uncannyEyes.ino`. Any design folder from that project drops straight into a face's `eye/`
  directory. The goat and dragon faces use their designs as published; the cat's is their dragon
  recoloured green, Clara's is the cat's iris recoloured ice blue, with a round pupil and her own
  almond lids, and the cyclops wears the same eye again in brown-red, alone and much larger.
  MIT licensed, by Phil Burgess and contributors.
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** and **[Vosk](https://alphacephei.com/vosk/)**
  for local speech recognition, **[Piper](https://github.com/OHF-Voice/piper1-gpl)** and
  **[Kokoro](https://github.com/thewh1teagle/kokoro-onnx)** for local voices,
  **[edge-tts](https://github.com/rany2/edge-tts)** for a free cloud voice,
  **[Ollama](https://ollama.com)** for a local brain, and **[pygame-ce](https://pyga.me/)** for
  the window and the audio clock.
- **Sound effects**: [Pixabay](https://pixabay.com/sound-effects/) is a good source, free to use
  and no attribution required. Drop the files into `faces/<name>/sounds/` and the file name
  becomes the sound's name. None are committed to this repository.

## Troubleshooting

**Mouth stuck on sil / not moving**
Check the debug overlay (`D`). If Time counts up but Viseme stays sil, no word
boundaries arrived (edge-tts must be called with `boundary="WordBoundary"`; this is
set in `talker/tts_backends.py`). If Time is stuck, the audio stream did not open — look for
`[audio]` lines in the terminal.

**Lips early or late**
Use `--sync-offset` (whole face vs audio, seconds) and `--lead` (how far mouth shapes
lead their sound). Start with `--debug` and adjust by 0.02 s steps.

**Mouth PNG in wrong position**
All mouth PNGs should be full 800x800 canvases. (They are cropped to their opaque
bounds at load time, so this is only a placement convention, not a performance cost.)

**ffmpeg not found**
Install `imageio-ffmpeg` (`pip install imageio-ffmpeg`) for a bundled binary, or
install ffmpeg system-wide and add it to PATH.

**PyAudio install fails**
Install the PortAudio dev package first (`portaudio19-dev` / `portaudio-devel` /
`brew install portaudio`). On Windows, `pip install pyaudio` has wheels.
You can run without any audio device using `--no-audio`.

**edge-tts fails / no audio**
Needs internet to reach Microsoft speech servers. Errors are printed as `[speech] TTS failed: ...`
and, with `--auto-exit`, the window closes instead of hanging.

**g2p-en says unavailable**
It needs two NLTK corpora; they are downloaded on first run. Without them the regex
fallback is used automatically (slightly less accurate mouth shapes).
