"""
setup_wizard.py — one command that gets a new machine ready.

    python voice_loop.py --setup

Everything it does was already possible one flag at a time. What it adds is
order, and checking things up front that otherwise fail late: a missing
Anthropic key is not noticed until the character first tries to answer, and
then it only says "Sorry, I could not think of an answer", which tells you
nothing. Here it is one line at the top of the run.

Seven steps, each printing a pass/fail line and carrying on where it can:

    1  packages and ffmpeg
    2  API keys, checked against the services rather than just present
    3  the speaker, tested by speaking through it
    4  the microphone, tested by showing you the level
    5  the camera, if you want one
    6  the room, via the existing three-measurement calibration
    7  a summary and the exact command to run

Nothing here is new machinery: it drives audio_engine, tts_backends, vision and
calibrate, and writes the same calibration.json the loop already reads.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, WARN, BAD, SKIP = "ok", "warn", "failed", "skipped"
_MARK = {OK: "  ok  ", WARN: " note ", BAD: "FAILED", SKIP: " skip "}


class Report:
    """What each step found, so the summary can be written once at the end."""

    def __init__(self):
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, state: str, what: str, detail: str = "") -> str:
        self.rows.append((state, what, detail))
        print(f"  [{_MARK[state]}] {what}{('  — ' + detail) if detail else ''}")
        return state

    @property
    def failed(self) -> List[Tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == BAD]


def _head(n: int, title: str) -> None:
    print(f"\n{n}. {title}\n" + "-" * (len(title) + 3))


def _ask(prompt: str, default: str = "") -> str:
    try:
        got = input(f"   {prompt}{f' [{default}]' if default else ''}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return got or default


def _yes(prompt: str, default: bool = True) -> bool:
    return _ask(prompt + (" [Y/n]" if default else " [y/N]")).lower()[:1] in (
        ("", "y") if default else ("y",))


# ── 1. packages ──────────────────────────────────────────────────────────────
def check_packages(report: Report) -> None:
    _head(1, "Packages and tools")
    for mod, why, needed in (("numpy", "maths everywhere", True),
                             ("pygame", "the window and the audio clock", True),
                             ("pyaudio", "the microphone and speaker", True),
                             ("faster_whisper", "the default speech recognizer", True),
                             ("cv2", "the camera (only with --camera)", False),
                             ("anthropic", "the Claude brain", False)):
        try:
            __import__(mod)
            report.add(OK, mod, why)
        except ImportError:
            hint = "pip install -r requirements.txt"
            if mod == "pyaudio":
                hint = "needs PortAudio first: dnf install portaudio-devel (or apt install portaudio19-dev)"
            report.add(BAD if needed else WARN, f"{mod} is missing", f"{why}; {hint}")

    from .tts_backends import find_ffmpeg
    if find_ffmpeg():
        report.add(OK, "ffmpeg", "only the free edge voice needs it")
    else:
        report.add(WARN, "ffmpeg not found",
                   "fine unless you use --tts edge; install it or pip install imageio-ffmpeg")

    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    report.add(OK if free_gb > 3 else WARN, f"{free_gb:.1f} GB free",
               "speech and voice models download to ~/.cache/talker on first use")


# ── 2. keys ──────────────────────────────────────────────────────────────────
def _check_anthropic(key: str) -> Tuple[str, str]:
    try:
        from .brains.claude_chat import make_client
        make_client().messages.create(model="claude-haiku-4-5", max_tokens=1,
                                      messages=[{"role": "user", "content": "hi"}])
        return OK, "accepted"
    except Exception as e:
        msg = str(e)
        if "credit" in msg.lower() or "billing" in msg.lower():
            return BAD, "key works but the account has no credit; add some under Plans & Billing"
        return BAD, msg.splitlines()[0][:120]


def _check_http(urls, headers: dict) -> Tuple[str, str]:
    """Try each endpoint; one success is enough.

    A key can be scoped to just the permissions it needs, and then the obvious
    "who am I" endpoint returns 401 while the key is perfectly good for what we
    actually do with it. Reporting that as a dead key sends people hunting for a
    problem they do not have.
    """
    import urllib.error
    import urllib.request
    reachable = False
    for url in ([urls] if isinstance(urls, str) else urls):
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15).read(1)
            return OK, "accepted"
        except urllib.error.HTTPError as e:
            reachable = True
            if e.code not in (401, 403):
                return BAD, f"HTTP {e.code} from {url}"
        except Exception:
            continue
    if reachable:
        return BAD, "the service rejected the key on every endpoint tried"
    return WARN, "could not reach the service; check the network"


def _check_elevenlabs(key: str) -> Tuple[str, str]:
    """An ElevenLabs key is usually scoped, and a speech-only key refuses every
    read endpoint. So ask the read endpoints first, for free, and if they all
    refuse, synthesise a single character, which is what we use the key for and
    costs one character of quota."""
    state, detail = _check_http(["https://api.elevenlabs.io/v1/models",
                                 "https://api.elevenlabs.io/v1/voices",
                                 "https://api.elevenlabs.io/v1/user"], {"xi-api-key": key})
    if state == OK:
        return OK, detail
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?output_format=mp3_22050_32",
        data=b'{"text":".","model_id":"eleven_flash_v2_5"}',
        headers={"xi-api-key": key, "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=20).read(1)
        return OK, "accepted for speech (the key is scoped: no account or voice reads)"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:100]
        if e.code in (401, 403):
            return BAD, "rejected for speech too, so the key is wrong or revoked"
        return WARN, f"speech returned HTTP {e.code}: {body}"
    except Exception as e:
        return WARN, f"could not reach the service ({e})"


def check_keys(report: Report) -> None:
    _head(2, "API keys")
    from .env_config import load_dotenv
    load_dotenv()
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        report.add(WARN, "no .env file", "keys can also come from the environment")
        if _yes("Create .env from .env.example now?"):
            shutil.copy(os.path.join(ROOT, ".env.example"), env_path)
            print(f"   wrote {env_path} — put your keys in it and run --setup again")

    checks = (("ANTHROPIC_API_KEY", "Claude, the default brain", True,
               lambda k: _check_anthropic(k)),
              ("ELEVENLABS_API_KEY", "ElevenLabs voice and speech to text", False,
               _check_elevenlabs),
              ("OPENAI_API_KEY", "OpenAI, for comparison only", False,
               lambda k: _check_http("https://api.openai.com/v1/models", {"Authorization": f"Bearer {k}"})))

    for name, why, required, probe in checks:
        key = os.environ.get(name, "")
        if not key or key.endswith("..."):
            report.add(BAD if required else SKIP, f"{name} not set", why)
            continue
        print(f"   checking {name} ...", end=" ", flush=True)
        state, detail = probe(key)
        print()
        report.add(state, name, detail)


# ── 3 & 4. devices ───────────────────────────────────────────────────────────
def _pick(kind: str, devices: List[tuple]) -> Optional[str]:
    if not devices:
        return None
    print(f"   {kind} devices:")
    for idx, name, rate, is_default in devices:
        print(f"     [{idx}] {name}  ({rate} Hz){'  <- system default' if is_default else ''}")
    default = next((str(i) for i, _, _, d in devices if d), str(devices[0][0]))
    got = _ask(f"Which {kind}? number, or a name fragment, or blank for the default", default)
    return got or None


def check_output(report: Report, audio) -> Optional[str]:
    _head(3, "Speaker")
    choice = _pick("output", audio.list_output_devices())
    if choice is None:
        report.add(BAD, "no output devices", "the character will be silent")
        return None
    try:
        audio.close_output()                       # reopen on the chosen one
        audio.output_device = audio.resolve_device(choice, "output")
    except Exception as e:
        report.add(BAD, f"output {choice!r} could not be selected", str(e)[:120])
        return None
    try:
        from .tts_backends import make_backend
        from .phoneme_scheduler import ScheduleReader
        from .speech_pipeline import SpeechPipeline
        backend = make_backend("piper" if _have_piper() else "edge")
        pipe = SpeechPipeline(audio, ScheduleReader(), backend)
        pipe.start()
        print("   speaking a test line ...")
        pipe.speak("Setup test. If you can hear this, the speaker is working.")
        t0 = time.monotonic()
        while (pipe.is_busy or time.monotonic() - t0 < 2) and time.monotonic() - t0 < 25:
            time.sleep(0.1)
        pipe.stop()
        report.add(OK if _yes("Did you hear it?") else BAD, f"output {choice!r}")
    except Exception as e:
        report.add(BAD, f"output {choice!r} could not speak", str(e)[:120])
    return choice


def _have_piper() -> bool:
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def check_mic(report: Report, audio) -> Optional[str]:
    _head(4, "Microphone")
    choice = _pick("input", audio.list_input_devices())
    if choice is None:
        report.add(BAD, "no input devices", "nothing to listen with")
        return None
    try:
        audio.start_mic(on_frames=None, rate=16000, device=choice)
    except Exception as e:
        report.add(BAD, f"microphone {choice!r} would not open", str(e)[:160])
        return None
    print("   Talk now, from where a visitor would stand. Aim for 500 to 5,000.")
    peak = 0.0
    for _ in range(60):                       # six seconds
        time.sleep(0.1)
        peak = max(peak, audio.get_state()[0])
        print(f"\r   level {audio.get_state()[0]:6.0f}   peak {peak:6.0f}", end="", flush=True)
    print()
    if peak < 60:
        report.add(BAD, f"microphone {choice!r} heard nothing",
                   f"peak {peak:.0f}. Either nobody spoke, or this is the wrong device")
    elif peak < 200:
        report.add(BAD, f"microphone {choice!r} is far too quiet", f"peak {peak:.0f}; raise the gain")
    elif peak < 500:
        report.add(WARN, f"microphone {choice!r} is quiet", f"peak {peak:.0f}; raise the gain a little")
    elif peak > 24000:
        report.add(WARN, f"microphone {choice!r} is too hot", f"peak {peak:.0f}; lower the gain to avoid clipping")
    else:
        report.add(OK, f"microphone {choice!r}", f"peak {peak:.0f}")
    return choice


# ── 5. camera ────────────────────────────────────────────────────────────────
def check_camera(report: Report) -> Optional[str]:
    _head(5, "Camera (optional)")
    if not _yes("Will this character use a camera?", default=False):
        report.add(SKIP, "camera", "run without --camera")
        return None
    try:
        from .vision import CameraSource, list_cameras, resolve_camera
        cams = list_cameras()
        if not cams:
            report.add(WARN, "no cameras found")
            return None
        for c in cams:
            print(f"     [{c['index']}] {c['name']}  ({c['path']})")
        choice = _ask("Which camera? number or a name fragment", str(cams[0]["index"]))
        src = CameraSource(resolve_camera(choice))
        frames = src.burst(1)
        src.close()
        report.add(OK if frames else BAD, f"camera {choice!r}",
                   f"{len(frames[0]) // 1024} KB frame" if frames else "no frame came back")
        return choice
    except Exception as e:
        report.add(BAD, "camera", str(e)[:140])
        return None


# ── 6. the room ──────────────────────────────────────────────────────────────
def check_room(report: Report, audio, mic: Optional[str], out: Optional[str]) -> None:
    _head(6, "The room (optional)")
    if not (mic and out):
        report.add(SKIP, "calibration", "needs a working mic and speaker")
        return
    if not _yes("Measure the room now? Takes about a minute and sets barge-in."):
        report.add(SKIP, "calibration", "run --calibrate later")
        return
    try:
        from .calibrate import run_calibration
        from .phoneme_scheduler import ScheduleReader
        from .speech_pipeline import SpeechPipeline
        from .tts_backends import make_backend
        pipe = SpeechPipeline(audio, ScheduleReader(), make_backend("piper" if _have_piper() else "edge"))
        pipe.start()
        try:
            res = run_calibration(audio, pipe.speak, lambda: pipe.is_busy, mic, out)
        finally:
            pipe.stop()
        report.add(OK if res.get("barge_in_ok") else WARN, "calibration written to calibration.json",
                   res.get("verdict", ""))
    except Exception as e:
        report.add(BAD, "calibration", str(e)[:140])


# ── 7. summary ───────────────────────────────────────────────────────────────
def device_name(audio, choice: Optional[str], kind: str) -> Optional[str]:
    """Turn whatever was typed into a name fragment.

    PortAudio renumbers devices whenever anything is plugged in, so an index is
    the wrong thing to write into a command you will reuse tomorrow.
    """
    if not choice:
        return None
    if not str(choice).strip().isdigit():
        return str(choice)
    devices = audio.list_input_devices() if kind == "input" else audio.list_output_devices()
    for idx, name, _rate, _default in devices:
        if idx == int(choice):
            return name
    return str(choice)


def save_devices(mic: Optional[str], out: Optional[str]) -> bool:
    """Merge the chosen devices into calibration.json without losing measurements."""
    import json
    from .calibrate import CALIBRATION_FILE
    data = {}
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    if mic:
        data["mic_device"] = mic
    if out:
        data["output_device"] = out
    data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def summarise(report: Report, mic, out, cam) -> int:
    _head(7, "Summary")
    for state, what, detail in report.rows:
        print(f"  [{_MARK[state]}] {what}{('  — ' + detail) if detail else ''}")
    cmd = ["python voice_loop.py --face clara"]
    if mic:
        cmd.append(f'--mic-device "{mic}"')
    if out:
        cmd.append(f'--output-device "{out}"')
    if cam:
        cmd.append(f'--camera "{cam}"')
    cmd.append("--borderless")
    print("\nRun it with:\n\n    " + " ".join(cmd) + "\n")
    if save_devices(mic, out):
        print("Those devices are saved in calibration.json, so the two device flags are\n"
              "optional next time. Any flag you do pass still wins.")
    if report.failed:
        print(f"\n{len(report.failed)} thing(s) still need fixing before that will work.")
        return 1
    return 0


def run_setup(args) -> int:
    print("Talker setup. Enter accepts the suggestion in brackets; Ctrl-C stops.\n")
    report = Report()
    check_packages(report)
    check_keys(report)
    from .audio_engine import AudioEngine
    audio = AudioEngine()
    mic = out = cam = None
    try:
        out = check_output(report, audio)
        mic = check_mic(report, audio)
        cam = check_camera(report)
        check_room(report, audio, mic, out)
        mic = device_name(audio, mic, "input")       # names outlive indices
        out = device_name(audio, out, "output")
    finally:
        try:
            audio.close()
        except Exception:
            pass
    return summarise(report, mic, out, cam)
