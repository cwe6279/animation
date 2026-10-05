# Talker on a Raspberry Pi Zero 2 W

The main README targets a Pi 5 (4 GB+). This page covers the Pi Zero 2 W:
512 MB of RAM, four slow Cortex-A53 cores, one micro-USB data port, and Wi-Fi
and Bluetooth sharing a single radio. Talker runs on it if every heavy stage
(speech-to-text, brain, voice) is in the cloud and the face is kept light.

Measured on a Zero 2 W running Debian 13 (trixie), Python 3.13, October 2026.

## What runs where

| Stage | On the Zero 2 W | Why |
|---|---|---|
| Speech-to-text | ElevenLabs Scribe (cloud) | Local Whisper/Vosk is too slow or too inaccurate here |
| Brain | Claude Haiku (cloud) | No room for a local model |
| Voice | ElevenLabs Flash (cloud) | Raw PCM, no ffmpeg decode; edge-tts works as a free fallback |
| Face | pygame on the Pi | The only stage that has to run locally; see [Face frame rate](#face-frame-rate) |

## Hardware

- **Power:** a 5 V / 2.5 A supply (the official Raspberry Pi micro-USB supply) into the **PWR** port. A laptop USB port (0.5–0.9 A) runs light tests, but the Pi reset without warning during a four-core faster-whisper run while powered from a laptop.
- **Pi Zero 2 W** with a 16 GB+ SD card. An 8 GB card ends up about 83% full after this install.
- **Microphone:** a USB mic or USB audio adapter on a micro-USB OTG adapter or hub, plugged into the port marked **USB**, not **PWR**. A Bluetooth speaker's built-in mic does not work out of the box; see [Known issues](#known-issues).
- **Speaker:** a USB audio adapter is the most reliable. A Bluetooth (A2DP) speaker works too; see [Bluetooth speaker](#bluetooth-speaker). HDMI audio works if the display has speakers.
- **Display:** an HDMI projector or monitor through a mini-HDMI adapter.
- **Camera (optional):** a Camera Module 3 (imx708) is detected. Leave vision off until the rest is tuned.

## Install

Debian/Raspberry Pi OS trixie already ships ffmpeg and git. Use the apt builds
of PyAudio and numpy so nothing has to compile on the Zero, and install only
the Python packages the cloud setup needs. The full `requirements.txt` pulls in
faster-whisper, piper, onnxruntime and opencv: over 1 GB, and none of it is
needed here. Those imports are lazy, so leaving them out is safe.

```bash
sudo apt install -y portaudio19-dev python3-pyaudio python3-numpy python3-venv pipewire-alsa
git clone https://github.com/cwe6279/animation.git ~/animation
cd ~/animation
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --prefer-binary pygame-ce edge-tts anthropic g2p-en pytest
pytest tests        # expect 1 failure: test_edge_live (see Known issues)
```

`pipewire-alsa` is required. Without it, ALSA's `default` device points only at
HDMI. PortAudio then finds no output device at all (`OSError: No Default
Output Device Available`), even though `pw-play` reaches a Bluetooth speaker
fine.

### Keys

Create `~/animation/.env`, then lock it with `chmod 600 ~/animation/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=sk_...
```

An ElevenLabs key needs the **Text to Speech** and **Speech to Text**
permissions (elevenlabs.io → Developers → API Keys). Without them TTS fails
with `missing the permission text_to_speech` and Scribe returns `unauthorized`.

## Bluetooth speaker

The Pi boots with every radio soft-blocked (`/etc/modprobe.d/rfkill_default.conf`)
and restores the last saved state. Unblock Bluetooth once and the saved state
keeps it unblocked across reboots. None of the steps below need sudo.

```bash
/usr/sbin/rfkill unblock bluetooth
bluetoothctl power on
```

Pair inside a single `bluetoothctl` session. Separate invocations forget the
scan results and fail with `Device ... not available`. Put the speaker in
pairing mode first:

```bash
bluetoothctl
  agent NoInputNoOutput
  default-agent
  scan on            # wait for the speaker's name, note its MAC
  pair  XX:XX:XX:XX:XX:XX
  trust XX:XX:XX:XX:XX:XX
  connect XX:XX:XX:XX:XX:XX
  quit
pw-play /usr/share/sounds/alsa/Front_Center.wav
```

Being trusted only lets the speaker connect to the Pi. After a reboot, the Pi
won't call the speaker back on its own. A small user service keeps it
connected: at login, after drops, and after the speaker is switched off and on.

`~/.local/bin/bt-speaker-keepalive` (chmod +x):

```bash
#!/bin/bash
CONF="$HOME/.config/talker/bt-speaker"
while true; do
    MAC=$(tr -d '[:space:]' < "$CONF" 2>/dev/null)
    if [ -n "$MAC" ]; then
        bluetoothctl power on >/dev/null 2>&1
        if ! bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
            bluetoothctl connect "$MAC" >/dev/null 2>&1 && echo "connected $MAC"
        fi
    fi
    sleep 10
done
```

`~/.config/systemd/user/bt-speaker.service`:

```ini
[Unit]
Description=Keep the Talker Bluetooth speaker connected
After=pipewire.service wireplumber.service
Wants=wireplumber.service

[Service]
ExecStart=%h/.local/bin/bt-speaker-keepalive
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
mkdir -p ~/.config/talker
echo XX:XX:XX:XX:XX:XX > ~/.config/talker/bt-speaker
systemctl --user daemon-reload
systemctl --user enable --now bt-speaker.service
```

Tested: after a deliberate disconnect the speaker reconnects within about
15 seconds, and after a full reboot it is connected and set as the default
output. This relies on the desktop auto-login, which starts PipeWire and the
user services. For a headless setup without auto-login, enable linger:
`sudo loginctl enable-linger $USER`.

A Bluetooth speaker adds roughly 150–250 ms of delay (A2DP with the SBC
codec). Compensate with `--sync-offset` (see [Tuning](#tuning)). On the Zero
2 W, Wi-Fi and Bluetooth share one radio, so streaming audio can slow the
Wi-Fi that every cloud call depends on.

## Benchmarks

Measured on the Zero 2 W over home Wi-Fi.

### Brain, voice and speech-to-text

| Measurement | Result |
|---|---|
| Claude Haiku, time to first token | ~470–500 ms |
| Haiku tag compliance (`tools/bench_llm.py`) | 100% known tags, 100% leading, 0% markdown, ~35 words |
| ElevenLabs Scribe accuracy (`tools/bench_stt.py --synth`) | 0.7% word error rate (0.0% clean, 1.4% noisy), 0.59 s per clip |
| Question to first audio, Haiku + Flash (`tools/bench_e2e.py --pairs cloud`) | **~1.0 s** median (the first request of a run takes ~5 s to warm up) |

### Voice: time to first audio through the real pipeline

| Voice | Median | Notes |
|---|---:|---|
| ElevenLabs Flash | **405 ms** | Fastest. Emotion tags are removed, not performed. |
| ElevenLabs v3 | 792 ms | Performs emotion tags |
| edge-tts (free) | 859 ms | The first line of a run takes ~2.7 s while ffmpeg starts up. Before warm-up it measured 8 s. |

Peak memory for the voice pipeline was 161 MB; about 235 MB is free with the desktop running.

### Face frame rate

Per-frame cost measured offscreen (update + draw + scale to 720p):

| Face | Eyes | Frame time | Max fps |
|---|---|---:|---:|
| ghost | drawn in code | 8 ms | 122 |
| eve | still images | 50 ms | 20 |
| skull | still images | 51 ms | 19 |
| cat, goat, dragon | textured live eyes | 91–93 ms | 11 |
| clara (1280×720) | textured live eyes | 91 ms | 11 |
| pumpkin (1280×720) | glow effects drawn in code | 121 ms | 8 |

`--profile pi` asks for 30 fps, and the adaptive governor only steps down as
far as 15. On the Zero 2 W only the cheap faces reach that. A textured eye
costs about 40 ms per eye per frame. The CPU was not throttling (42 °C,
`get_throttled=0x0`), so this is simply the CPU's limit.

## Tuning

Start from the Pi profile and override as needed:

```bash
cd ~/animation && . .venv/bin/activate
TALKER_FPS=20 python voice_loop.py --profile pi --face eve --tts elevenlabs --tts-model flash \
    --mic-device <usb mic fragment> --web-host 0.0.0.0
```

- **Frame rate:** set `TALKER_FPS` to a rate the face can actually hold: about 20 for faces with still-image eyes, about 12 for textured-eye faces. Use `--fixed-fps` to stop the governor stepping up and down.
- **Heavy faces:** for textured-eye faces, replace `textured_eye` with still-image eyes in `face.json`, or use a smaller canvas and let fullscreen scale it up.
- **Voice:** use ElevenLabs Flash for speed. Use v3 when acting out the emotion tags is worth about 0.4 s more.
- **Lip sync over Bluetooth:** start with `--sync-offset 0.2` and adjust in 0.02 s steps until the mouth matches the sound.
- **Mic gain:** a Samson Go Mic starts at its maximum hardware gain (+22 dB), which clips speech at arm's length and raises the room background to about 1,000. Set `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.5` (+3 dB): speech then reads 1,700–5,800 in `--mic-test`, the background about 100, and transcripts are exact. WirePlumber remembers the setting across reboots. Run `wpctl set-default <id>` first if a `bluez_input` stub is still the default input.
- **Background speech:** in `--mic-test`, a TV in the room is transcribed as confident, wrong sentences. In half-duplex mode `--mic-test --play` captures only the room between the character's lines, never the character's own voice. Rely on wake words in a noisy room.
- **End of turn:** the profile sets `--silence-ms 500`. Lower it for snappier replies, raise it if visitors get cut off mid-sentence.
- **Settings page:** `http://<pi-ip>:8020` while `voice_loop.py` is running. It's only reachable from other machines with `--web-host 0.0.0.0`.
- **Memory:** booting to the console instead of the desktop frees about 100 MB. Keep VS Code Remote-SSH disconnected while the face runs, because its server uses 200–300 MB.

## Known issues

- **`tests/test_edge_live.py` fails:** it asserts that the first audio arrives within 2 s. On the Zero, the first edge-tts line takes 2.7–8 s while ffmpeg starts up.
- **Bluetooth headset mode is silent:** switching a speaker to `headset-head-unit` connects with mSBC, but the speaker plays nothing and the mic records exact digital silence (peak 0). By default the Pi's Broadcom chip sends headset (SCO) audio to its unconnected hardware pins instead of to the software. A vendor command (`hcitool cmd 0x3f 0x01c 0x01 0x02 0x00 0x01 0x01`, as root, at every boot) can re-route it, but this hasn't been tested here. Even when it works, headset mode means call-quality 16 kHz mono playback and no interrupting mid-reply. Use a USB mic and keep the speaker in `a2dp-sink`.
- **Clipped syllables with edge-tts:** the first few milliseconds of edge-tts lines can be cut off over Bluetooth. Not yet diagnosed.
- **pip warnings during install:** piwheels prints pages of "Wheel filename ... is not correctly normalised" warnings while pip resolves packages. They are harmless.
- **sudo needs a password:** steps that need root (apt, reboot) need an interactive session. Everything in the Bluetooth section runs as the normal user.
