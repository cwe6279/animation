# A satellite: the ESP32-S3 AMOLED board as ears, voice and face

The board cannot run talker (no Linux, no Python, 8 MB of RAM). What it can do well is
the edge of the loop: wake word, microphone, speaker, a face on its screen, a battery.
So the board runs the open [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)
firmware unchanged, and talker gains a small server that speaks its protocol:

```
board  --wake word, Opus 16 kHz-->  talker/satellite.py  --> STT --> brain --> TTS
board  <--Opus 24 kHz, events-----  (tts start/stop, subtitles, stt text, emotion)
```

Everything else is the usual voice loop: Clara's memory, errands, notes and the local
face window all keep working. The host can be the desktop or a Pi.

```
python voice_loop.py --face clara --satellite --tts piper
```

`--satellite` replaces the sound device with the board (`SatelliteAudioEngine`) and
starts the server on port 8030 (`--satellite-port`, `--satellite-host`). The board
learns the WebSocket address by asking `http://<host>:8030/xiaozhi/ota/` at boot, so
the only thing it needs to know is that URL. `http://<host>:8030/` shows the device
that last checked in and whether it is connected.

Protocol: `docs/websocket.md` in the firmware repo. Summary: the board opens a
WebSocket after its wake word, sends `hello`, streams Opus (16 kHz mono, 60 ms) while
listening; we answer with `tts start`, Opus frames (24 kHz, so TTS audio is not
resampled), `tts sentence_start` for its subtitle line, `llm emotion` for its face,
`tts stop`. Then it listens again; after 40 s (`--satellite-idle`) with nothing said we close the channel
and it goes back to waiting for the wake word. The board cannot be woken from the
server side, so an announcement while it is idle is dropped (noted in the log).

## The board

Waveshare **ESP32-S3-Touch-AMOLED-1.8 V2**, sold by Spotpear with a "DeepSeek" label.
ESP32-S3, 8 MB PSRAM, 16 MB flash, 368x448 AMOLED (SH8601), CST816S touch, ES8311
codec with mic and speaker, AXP2101 power chip with a battery, QMI8658 IMU. Stock
firmware: Waveshare's Brookesia launcher with a Xiaozhi app locked to the public
Tenclass server (`~/esp32-backups/brookesia-stock-16MB.bin` is a full backup).

Lessons that cost an afternoon:

- **Only a USB-C to USB-C cable worked.** Two USB-A to C cables powered the board
  but it never enumerated. It shows up as `303a:1001 Espressif USB JTAG/serial`
  on `/dev/ttyACM0`.
- **It has a battery, so unplugging does not reset it.** Download mode is: hold PWR
  until the screen is dark, hold BOOT, tap PWR, release BOOT after two seconds.
- **A reset over USB re-enumerates the port**, so a serial reader opened before the
  reset dies. Reset with `esptool --after watchdog-reset`, wait for the port to come
  back, then open it with DTR and RTS both asserted (otherwise opening the port drops
  the chip into download mode).
- **Port permissions:** a udev rule, since the node is recreated on every reset:
  `SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", MODE="0666"` in
  `/etc/udev/rules.d/99-espressif.rules`.
- **V2, not V1.** The V1 board definition uses an FT5x06 touch driver and crashes in
  a reboot loop on this board (black screen). The stock firmware's strings showed a
  CST816S driver, which is what the V2 definition uses.

## Building and flashing

The firmware needs ESP-IDF 6 (`release/v6.1`), about 3 GB with the toolchain:

```
mkdir -p ~/esp && cd ~/esp
git clone --depth 1 -b release/v6.1 --recursive --shallow-submodules https://github.com/espressif/esp-idf.git
./esp-idf/install.sh esp32s3
git clone --depth 1 https://github.com/78/xiaozhi-esp32.git
```

Point the default server at the talker host (saves a trip through the board's setup
page), then build for the board:

```
cd ~/esp/xiaozhi-esp32
echo 'CONFIG_OTA_URL="http://192.168.68.79:8030/xiaozhi/ota/"' >> sdkconfig.defaults
source ~/esp/esp-idf/export.sh
python scripts/build.py waveshare/esp32-s3-touch-amoled-1.8-v2
```

First flash (erases everything, including the saved Wi-Fi):

```
esptool --chip esp32s3 --port /dev/ttyACM0 erase-flash
esptool --chip esp32s3 --port /dev/ttyACM0 -b 921600 --after watchdog-reset \
    write-flash --flash-mode dio --flash-size 16MB --flash-freq 80m 0x0 build/merged-binary.bin
```

Later flashes: write only the application so the Wi-Fi settings survive (the merged
image pads over the settings area and the board asks for Wi-Fi again):

```
esptool --chip esp32s3 --port /dev/ttyACM0 -b 921600 --after watchdog-reset \
    write-flash --flash-mode dio --flash-size 16MB --flash-freq 80m 0x20000 build/xiaozhi.bin
```

Wi-Fi setup, once: the board opens a hotspot named `Xiaozhi-XXXX`; join it from a
phone, open `192.168.4.1`, pick the network, enter the password. The same page has an
advanced section with the server address, for a board that was not built with it.
Pressing BOOT during the first seconds after power-on brings that page back.

## Clara's face on the board (firmware fork, branch `clara-face`)

The fork lives in `~/esp/xiaozhi-esp32` on branch `clara-face`. What it adds:

- `main/boards/waveshare/esp32-s3-touch-amoled-1.8-v2/clara_face_display.*`: her eyes and
  mouth as LVGL image objects on black. The eyes are layers rendered on the host from the
  same textured-eye model talker uses (`faces/clara/eye`): a dark sclera, the iris disc,
  and five lid positions that also mask outside the eye. The iris wanders (gaze), the lids
  blink, the whole eye pair shifts a little with the emotion. All drawing happens in the
  LVGL task on a 20 ms timer; the audio task only records which frame is playing.
- A six-byte header the satellite server puts on each audio frame when the board's hello
  says `"clara_face": true`: magic `0xCF`, version, emotion id, then one mouth id per
  20 ms. The firmware carries it in the packet timestamp (bit 31 set) and fires
  `Display::SetFaceFrame` as the frame reaches the codec, so lips follow playback.
- The lids carry the expression: the same cuts as `EMOTION_LIDS` in the Python face
  (upper cover, lower cover, tilt, arch, corner drop) rendered per emotion for each eye
  and shipped as alpha-only masks, stacked under the blink mask. The server's emotion
  names map in `SetEmotion` (talker's "annoyed" travels as "confused").
- Partial redraws are aligned to even pixels (`LV_EVENT_INVALIDATE_AREA`); this panel
  puts odd-aligned patches a pixel off, which looked like the face jerking.
- Kconfig: `CLARA_FACE_SUBTITLES` (default off) shows what was heard and said in a line
  at the bottom; `WAKE_WORD_WHILE_SPEAKING` (off for Clara) is upstream's barge-in on the
  wake word, which cut her off on any noise that sounded like "Hi ESP".
- `CONFIG_OTA_URL` defaults to the talker host, and `PROJECT_VER` is bumped (2.5.10x) so
  the board accepts the fork over the air.

Regenerating the art (scale 0.6, positions from face.json) is the scratch script in this
session's notes; the C arrays are the LVGL `LVGLImage.py` output (RGB565 for mouths,
RGB565A8 for eye layers) dropped into the board folder, which CMake globs.

Each time the board connects, the server calls its `self.get_device_status` tool and
hands Clara one line of context: battery level and charging state, Wi-Fi signal, chip
temperature. `http://<host>:8030/` shows the raw status too.

Server-side knobs: `--satellite-volume 80` sets the board's speaker on connect through
its MCP tool (95 distorts); `--satellite-firmware build/xiaozhi.bin` offers that image
over the air.

The built-in speaker is tiny (12 x 10 mm, 8 ohm, 1 W on header H2, driven by an NS4150B
from 3.3 V, so about 0.6 W at most): quiet at 80, and above that the shell buzzes. Bass is
what shakes it, so the server shapes her voice for it before sending (`SpeakerShaper`): a
high-pass at `--satellite-bass-cut` (300 Hz by default; 400 sounded worse, 0 keeps the
bass), a `--satellite-boost` of 6 dB, and a limiter holding peaks at `--satellite-ceiling`
0.5 of full scale, which is what stopped the loud syllables from buzzing. That is as far as
software goes; the real fix is a larger 8 ohm speaker (1 to 2 W, as big as fits, in its own
small enclosure) on that header, or playing her voice from the host instead.

The mic is the opposite problem: quiet and very clean (room noise reads about 33 RMS on the
int16 scale where a desktop mic reads 200). Three host-side defaults change with
`--satellite` so a whisper gets through: the recorder opens at `--mic-min-rms` 120 instead
of 350; Whisper's Silero voice detector is off (`--stt-vad off`), since it discards
whispered speech outright, and the confidence filters do the noise rejection; and the
model is small.en. Each utterance is also lifted to a fixed peak before recognition.
`TALKER_STT_DUMP=<dir>` saves every clip Whisper sees, and the log says why a clip produced
nothing (level, what the detector kept, segments dropped), which is how this was found.
The firmware's mic gain is 30 dB (`input_gain_` in `es8311_audio_codec.cc`; the chip also
has 36 and 42) and was left alone.

Three timing rules the board imposes, each learned the hard way:

- **Pace the audio.** The board keeps about 1.2 s of incoming frames and silently drops
  the rest. A TTS backend hands over a whole sentence at once; the engine releases frames
  only 0.5 s ahead of their place on the timeline (`tests/test_satellite.py` checks it).
  Symptom of getting this wrong: every sentence longer than a second loses its tail.
- **Hold the first frames.** The board discards audio until its state machine has acted
  on `tts start`, so the first frames leave 150 ms after it. Otherwise the first half
  second of each reply is missing.
- **Mute the mic briefly after `listen start`.** The board's input buffers still hold the
  tail of her own voice; 400 ms of silence stops "Pick what?" turns where she answers
  herself. (AEC is off in this configuration.)

Wake word: `python scripts/build.py --wake-word custom waveshare/esp32-s3-touch-amoled-1.8-v2`
builds with the English speech-command model listening for `CONFIG_CUSTOM_WAKE_WORD`
(set in sdkconfig.defaults; `--wake-word custom` is a fork addition to the build script).
The assets partition must be flashed once (0x800000), since the model lives there, and
again whenever the phrase or threshold changes: the board reads both from the assets
config, not from the app. The stock detector only knows "Hi ESP".

The speech-command model is weaker than a trained wake-word model, and a single two-syllable
word is the hard case: a power-on log showed clear "Clara"s scoring 0.21 and 0.30 against a
0.20 threshold, and the first two of three were missed. So since 2.5.111 the option holds
several phrases separated by `;` (a fork addition to `build_default_assets.py`): spelling
variants that cover how the name is said ("clara;klara;clarra;claira") plus "hey clara" and
"hi clara", which a longer phrase makes reliable; any of them wakes her and all show as
"Clara". `CONFIG_CUSTOM_WAKE_WORD_THRESHOLD` is 15 (0.15; up for fewer false wakes), and the
model's recognition window is 8 s instead of 3 (it forgets what it heard at each reset, so a
word said across one was lost).

Battery behaviour (fork 2.5.109, only while discharging; on USB nothing below happens):
after `CONFIG_CLARA_SLEEP_SECONDS` (60) of idle the face stops and the panel is put to
sleep (DISPOFF then SLPIN via `ClaraFaceDisplay::SetPowerSaveMode`) while the wake word
keeps running; "Clara" brings the panel and the face back. After
`CONFIG_CLARA_SHUTDOWN_SECONDS` (240, counted from the start of idle) the board powers off
and only PWR wakes it. Idle starts when the host closes the channel, 40 s after her last
reply (`--satellite-idle`). Upstream's reduced-CPU sleep is not used: its code disables the
wake word. The timer is armed by the charger state, polled once a second: 2.5.108 armed it
at boot and only disarmed it on a battery-to-USB transition, so a board that booted on USB
switched itself off four minutes after its last reply. The board's status now carries the
battery voltage next to the percentage (the AXP2101's percentage is a voltage estimate and
falls fast from a 4.1 V charge), and the serial log says when and at what level it powered
off. Standby draw with the wake word running is the floor here: Wi-Fi is already in maximum
modem sleep when idle, the codec output and amplifier switch off after a few seconds of
silence, and the CPU cannot clock down without stopping the wake word.

Her eyes stay shut from power-on until the wake word is actually listening (Wi-Fi, the
version check, loading the model), then open once: a face with open eyes means "say
Clara".

Touch (fork 2.5.117): the panel's touch controller feeds the face. A tap is a glance at the
finger and a "Hm?"; a hold past 350 ms is patience and an "Mmm"; a rub or three quick taps is
a tickle; a two-second hold closes her eyes and blacks the screen, and a tap, the wake word or
USB wakes her. That sleep leaves the panel driver awake on purpose: with the panel in SLPIN
the touch controller reports nothing, so a tap could never wake her (found with the serial
log: no touch events at all while dark), and a black AMOLED draws next to nothing anyway.
The battery timer's deep panel sleep stays, where the wake word and USB are the wakers. The reactions are a temperament: `CLARA_TOUCH_COMPOSED` (default, Clara: the
tickle is a half-lidded, faintly annoyed look up) or playful (wide eyes, a contented squint,
a ha-ha with a giggle). The three clips are in her voice, made with ElevenLabs v3 and its
audio tags, trimmed, and encoded as 48 kHz mono Opus in OGG (`-frame_duration 60`) into
`main/assets/common/`, which the build embeds and names (`Lang::Sounds::OGG_HM`). They play
from the board itself, only while idle and at most once per 8 s each, with the wake word
paused for 3.5 s around the clip: the first time she said "let's keep it professional" the
detector took her own voice for her name and opened a channel. Composed Clara's hold is
silent; her tickle line is a throat-clear and "Let's keep it professional.". A tap while the
panel sleeps on battery hands the power-save timer a wake, so the idle countdown restarts
rather than being bypassed. Next: the board opens the channel on a touch and sends one
sampled event, so the brain can react in words as well ("Event: ..."), never a stream.

## What is next

- The pupil does not dilate and the eyes do not react to emotion beyond a small shift.
- Tune the one-word wake threshold against false wakes in a normal room.
- Waking the board for announcements: the protocol is device-initiated, so this
  needs a firmware change (a server-pushed "notify" on a kept-open channel).
- A better speaker, or her voice from the host's speaker while the board stays mic and face.
- Standby draw on battery is dominated by Wi-Fi plus the wake-word model (roughly 80 to
  120 mA); a deeper idle would need the chip to drop Wi-Fi and reconnect on wake.
