"""
web_panel.py — a small control page for the running character.

On by default at http://<this box>:8020 (voice_loop.py --web-port, --no-web);
if that port is taken it walks up to the next free one and prints the URL.
Standard library only, one daemon thread, nothing on the audio or render path:
the page polls /api/state twice a second and the loop never waits for it.

What it shows and does:
  * status: face, ears, brain, voice, mic level, speaking / thinking / engaged,
    the last [turn] timing line, the last thing heard and said, vision stats
  * live tunables with their help text: silence gate, barge-in, idle timeout,
    vision interval and change filter, debug overlay ... applied at once
  * tests for the physical setup: mic level meter, speak a line through the
    speaker, send a line as a visitor, interrupt, a camera snapshot, and the
    calibration routine (room, speaker bleed, a person) with its verdict
  * the full flag reference (every --option with its help and current value)
    and the face.json fields with what they do
  * Wi-Fi (Raspberry Pi kiosks): see networks and join one through NetworkManager

Register what the page may touch from voice_loop.py: panel.tunable(...),
panel.action(...), panel.status_fn, panel.calibrator, panel.snapshot.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8020


@dataclass
class Tunable:
    name: str
    get: Callable[[], Any]
    set: Callable[[Any], None]
    help: str
    kind: str = "float"           # int | float | bool | str
    unit: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    flag: str = ""                # the command-line flag that sets the same thing at startup

    def coerce(self, raw: Any) -> Any:
        if self.kind == "bool":
            return str(raw).lower() in ("1", "true", "on", "yes")
        if self.kind == "int":
            v: Any = int(float(raw))
        elif self.kind == "float":
            v = float(raw)
        else:
            return str(raw)
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        return v


@dataclass
class Action:
    name: str
    fn: Callable[[str], str]      # takes the optional text field, returns a message
    help: str
    takes_text: bool = False


# ── face.json field help (the "?" text next to each manifest value) ────────────
FACE_FIELD_HELP = {
    "canvas_w": "Canvas width in pixels. Procedural shapes scale with it; set 1080 for a pixel-exact projector.",
    "canvas_h": "Canvas height.", "fps": "Render rate; 30 halves the work on a Pi. TALKER_FPS overrides.",
    "bg_color": "Background colour; keep black for projection.",
    "glow_color": "Halo colour around procedural shapes and the ambient glow behind body art.",
    "glow_intensity": "0 disables the glow, 1 normal, 2 intense.",
    "glow_style": "halo: soft light spills outward. inner: crisp cut edges lit from inside (pumpkin).",
    "core_color": "inner style: the hot spot colour (default: the shape colour pushed toward white).",
    "rim_color": "inner style: thin cut-edge line.", "light_offset": "inner style: moves the hot spot down.",
    "cut_depth": "inner style: [dx, dy] inset; the shell's inner wall shows on the other side.",
    "wall_color": "inner style: colour of that inner wall.",
    "eye_left": "cx, cy centre; image, scale, opacity for PNG eyes.", "eye_right": "Same for the right eye.",
    "eye_color": "Procedural eye colour.", "draw_eyes": "false: no eyes at all (a voice-only character).",
    "blink": "Blink on or off.", "blink_interval": "[min, max] seconds between blinks; emotions scale it.",
    "blink_speed": "Higher is snappier.", "eye_speech_pulse": "Eyes grow by this fraction when the mouth is wide open.",
    "eye_lids": "Image eyes without pupils: emotions as lid cuts, blinks close the lids.",
    "gaze_amount": "Idle glance distance in px.", "gaze_interval": "[min, max] seconds between glances.",
    "gaze_speed": "Easing rate of a glance.", "gaze_while_speaking": "0 locks the eyes forward while talking.",
    "textured_eye": "Live eyes composed from parts in a folder: pupil moves, dilates, lids track.",
    "mouth": "anchor_cx/cy, min_w/max_w, style toothed | rounded | grin, n_teeth, color, opacity (0 = no mouth).",
    "mouth_images": "viseme key to PNG; six images cover all twelve shapes.",
    "draw_nose": "Procedural triangle nose.", "nose": "Nose PNG placement.",
    "draw_stem": "Pumpkin stem.", "voices": "Default voice per backend: elevenlabs id, edge name, piper voice.",
    "tts": "This face's own voice backend when --tts is not given: piper (local), elevenlabs, edge, fish.",
    "tts_model": "ElevenLabs model for this face (v3 performs tags, flash is faster).",
    "voice_speed": "Speaking rate multiplier.", "wake_words": "Names that start a conversation in wake mode.",
    "sleep_words": "Short phrases that end it at once.", "sounds": "Folder of sound effects for {{sfx name}}.",
    "body": "moves: the movement names the brain may ask for with {{move name}}.",
    "character": "Personality text (from character.md).",
}


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def describe_parser(parser, args) -> List[Dict[str, Any]]:
    """Every --flag with its help text and the value in use this run."""
    rows = []
    for a in parser._actions:
        if not a.option_strings or a.dest in ("help",):
            continue
        h = a.help or ""
        try:                                   # argparse style: %(default)s, %% for a literal percent
            h = h % {"default": a.default}
        except (KeyError, TypeError, ValueError):
            h = h.replace("%%", "%")
        rows.append({"flags": ", ".join(a.option_strings), "help": h,
                     "default": None if a.default is None else str(a.default),
                     "value": None if args is None else str(getattr(args, a.dest, None))})
    return rows


# ── Wi-Fi through NetworkManager (Raspberry Pi OS and most Linux) ───────────────
class WifiControl:
    def __init__(self):
        self.available = shutil.which("nmcli") is not None

    def _run(self, *cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
        return subprocess.run(["nmcli", "-t", *cmd], capture_output=True, text=True, timeout=timeout)

    def status(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": "nmcli (NetworkManager) is not installed on this box"}
        devs = []
        r = self._run("-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status")
        for line in r.stdout.splitlines():
            parts = self._fields(line)
            if len(parts) >= 4 and parts[1] in ("wifi", "ethernet"):
                devs.append({"device": parts[0], "type": parts[1], "state": parts[2], "connection": parts[3]})
        return {"available": True, "devices": devs, "ip": lan_ip()}

    @staticmethod
    def _fields(line: str) -> List[str]:
        """nmcli -t separates fields with ':' and escapes a ':' inside a value as '\\:'."""
        return [f.replace("\\:", ":") for f in re.split(r"(?<!\\):", line)]

    def scan(self) -> List[Dict[str, Any]]:
        r = self._run("-f", "ACTIVE,SSID,SIGNAL,SECURITY", "device", "wifi", "list", "--rescan", "yes", timeout=40)
        seen: Dict[str, Dict[str, Any]] = {}
        for line in r.stdout.splitlines():
            parts = self._fields(line)
            if len(parts) < 4:
                continue
            active, ssid, signal, sec = parts[0], parts[1], parts[2], parts[3]
            if not ssid:
                continue
            row = {"ssid": ssid, "signal": int(signal or 0), "security": sec or "open", "active": active == "yes"}
            if ssid not in seen or row["signal"] > seen[ssid]["signal"]:
                seen[ssid] = row
        return sorted(seen.values(), key=lambda x: (-x["active"], -x["signal"]))

    def connect(self, ssid: str, password: str = "") -> str:
        cmd = ["device", "wifi", "connect", ssid]
        if password:
            cmd += ["password", password]
        r = self._run(*cmd, timeout=60)
        out = (r.stdout + r.stderr).strip()
        return out or ("connected" if r.returncode == 0 else f"failed ({r.returncode})")


# ── the panel ─────────────────────────────────────────────────────────────────
class WebPanel:
    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.tunables: Dict[str, Tunable] = {}
        self.actions: Dict[str, Action] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=80)
        self.status_fn: Callable[[], Dict[str, Any]] = lambda: {}
        self.parser = None
        self.args = None
        self.manifest = None
        self.snapshot: Optional[Callable[[], Optional[bytes]]] = None
        self.calibrator: Optional[Callable[[Callable[[str], None]], Dict]] = None
        self.pause: Callable[[bool], None] = lambda on: None
        self.wifi = WifiControl()
        self.calib: Dict[str, Any] = {"state": "idle", "prompt": "", "result": None, "error": ""}
        self._calib_go = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.started_at = time.time()

    # registration -------------------------------------------------------------
    def tunable(self, name: str, get, set, help: str, kind: str = "float", unit: str = "",
                lo: Optional[float] = None, hi: Optional[float] = None, flag: str = "") -> None:
        self.tunables[name] = Tunable(name, get, set, help, kind, unit, lo, hi, flag)

    def action(self, name: str, fn: Callable[[str], str], help: str, takes_text: bool = False) -> None:
        self.actions[name] = Action(name, fn, help, takes_text)

    def record(self, kind: str, text: str) -> None:
        self.events.append({"t": time.time(), "kind": kind, "text": str(text)[:400]})

    # lifecycle ----------------------------------------------------------------
    def start(self) -> Optional[str]:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):          # keep the console for the character
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: Any, code: int = 200) -> None:
                self._send(code, json.dumps(obj).encode())

            def do_GET(self):
                u = urlparse(self.path)
                try:
                    if u.path == "/":
                        self._send(200, panel.page().encode(), "text/html; charset=utf-8")
                    elif u.path == "/api/state":
                        self._json(panel.state())
                    elif u.path == "/api/reference":
                        self._json(panel.reference())
                    elif u.path == "/api/snapshot.jpg":
                        jpg = panel.snapshot() if panel.snapshot else None
                        if jpg:
                            self._send(200, jpg, "image/jpeg")
                        else:
                            self._json({"error": "no camera (start with --camera)"}, 404)
                    elif u.path == "/api/wifi":
                        st = panel.wifi.status()
                        if st.get("available") and parse_qs(u.query).get("scan"):
                            st["networks"] = panel.wifi.scan()
                        self._json(st)
                    else:
                        self._json({"error": "not found"}, 404)
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

            def do_POST(self):
                u = urlparse(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                except ValueError:
                    body = {}
                try:
                    if u.path == "/api/set":
                        self._json(panel.set(body.get("name", ""), body.get("value")))
                    elif u.path == "/api/action":
                        self._json(panel.run_action(body.get("name", ""), str(body.get("text", ""))))
                    elif u.path == "/api/calibrate":
                        self._json(panel.calibrate(body.get("step", "start")))
                    elif u.path == "/api/wifi/connect":
                        panel.record("wifi", f"joining {body.get('ssid', '')!r}")
                        self._json({"message": panel.wifi.connect(str(body.get("ssid", "")), str(body.get("password", "")))})
                    else:
                        self._json({"error": "not found"}, 404)
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        self._server = None
        for port in ([self.port] if self.port == 0 else range(self.port, self.port + 10)):
            try:
                self._server = ThreadingHTTPServer((self.host, port), Handler)
                break
            except OSError as e:
                print(f"[web] port {port} is taken ({e.strerror or e}); trying the next")
        if self._server is None:
            print(f"[web] panel not started: no free port near {self.port}")
            return None
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]          # port 0 = pick a free one (tests)
        self._thread = threading.Thread(target=self._server.serve_forever, name="web-panel", daemon=True)
        self._thread.start()
        url = f"http://{lan_ip()}:{self.port}"
        print(f"[web] control page at {url}  (also http://localhost:{self.port}; --no-web to disable)")
        return url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    # state --------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        tun = []
        for t in self.tunables.values():
            try:
                v = t.get()
            except Exception as e:
                v = f"? {e}"
            tun.append({"name": t.name, "value": v, "kind": t.kind, "unit": t.unit, "help": t.help,
                        "lo": t.lo, "hi": t.hi, "flag": t.flag})
        try:
            status = self.status_fn() or {}
        except Exception as e:
            status = {"error": f"{type(e).__name__}: {e}"}
        return {"status": status, "tunables": tun,
                "actions": [{"name": a.name, "help": a.help, "takes_text": a.takes_text} for a in self.actions.values()],
                "events": list(self.events)[-40:], "calibration": dict(self.calib),
                "has_camera": self.snapshot is not None, "wifi": self.wifi.available,
                "uptime_s": round(time.time() - self.started_at)}

    def reference(self) -> Dict[str, Any]:
        flags = describe_parser(self.parser, self.args) if self.parser is not None else []
        face = []
        if self.manifest is not None:
            for k, v in vars(self.manifest).items():
                if k.startswith("_"):
                    continue
                if k == "character":
                    v = (v[:160] + "…") if isinstance(v, str) and len(v) > 160 else v
                face.append({"field": k, "value": json.dumps(v, default=str)[:200], "help": FACE_FIELD_HELP.get(k, "")})
        return {"flags": flags, "face": face}

    def set(self, name: str, raw: Any) -> Dict[str, Any]:
        t = self.tunables.get(name)
        if t is None:
            return {"error": f"unknown tunable {name!r}"}
        v = t.coerce(raw)
        t.set(v)
        self.record("panel", f"{name} = {v}")
        return {"name": name, "value": t.get()}

    def run_action(self, name: str, text: str) -> Dict[str, Any]:
        a = self.actions.get(name)
        if a is None:
            return {"error": f"unknown action {name!r}"}
        msg = a.fn(text) or "ok"
        self.record("panel", f"{name}: {msg}"[:200])
        return {"message": msg}

    # calibration --------------------------------------------------------------
    def calibrate(self, step: str) -> Dict[str, Any]:
        if self.calibrator is None:
            return {"error": "calibration is not available in this run"}
        if step == "continue":
            self._calib_go.set()
            return dict(self.calib)
        if self.calib["state"] in ("running", "waiting"):
            return dict(self.calib)

        def ask(msg: str) -> None:
            self.calib.update({"state": "waiting", "prompt": msg.strip()})
            self._calib_go.clear()
            self._calib_go.wait(timeout=600)
            self.calib.update({"state": "running", "prompt": "Talk now, normal voice, 5 seconds"})

        def run() -> None:
            self.calib.update({"state": "running", "prompt": "Measuring the room: stay quiet for 4 s",
                               "result": None, "error": ""})
            self.pause(True)
            try:
                self.calib["result"] = self.calibrator(ask)
                self.calib.update({"state": "done", "prompt": ""})
            except Exception as e:
                self.calib.update({"state": "error", "error": f"{type(e).__name__}: {e}", "prompt": ""})
            finally:
                self.pause(False)
            self.record("calibration", self.calib.get("error") or (self.calib["result"] or {}).get("verdict", "done"))

        threading.Thread(target=run, name="calibration", daemon=True).start()
        return dict(self.calib)

    # page ---------------------------------------------------------------------
    def page(self) -> str:
        return PAGE.replace("{{PORT}}", str(self.port))


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Talker control</title>
<style>
:root{--bg:#f4f2ec;--panel:#fff;--ink:#1b1a1f;--mute:#6a6672;--line:#dbd6cc;--accent:#0e7c8c;--ok:#2e7d4f;--warn:#a8571a;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#0e1015;--panel:#161a22;--ink:#e8e4da;--mute:#9d99a6;--line:#2a2f3a;--accent:#3fc3d6;--ok:#6fd39a;--warn:#f0a16a;--bad:#ff7b72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "IBM Plex Sans","Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:baseline;gap:1rem;padding:1rem 1.4rem;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:1.15rem;margin:0}header .sub{color:var(--mute);font-size:.85rem}
nav{display:flex;gap:.2rem;padding:.4rem 1.4rem;border-bottom:1px solid var(--line);flex-wrap:wrap;background:var(--panel)}
nav button{background:none;border:0;padding:.45rem .8rem;border-radius:6px;color:var(--mute);font:inherit;cursor:pointer}
nav button.on{background:var(--bg);color:var(--ink);font-weight:600}
main{padding:1.2rem 1.4rem;max-width:1100px;margin:0 auto}section{display:none}section.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.8rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem}
.card h3{margin:0 0 .3rem;font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--mute)}
.big{font-size:1.25rem;font-weight:600}.mute{color:var(--mute)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.meter{height:10px;background:var(--bg);border:1px solid var(--line);border-radius:5px;overflow:hidden;margin-top:.4rem}
.meter i{display:block;height:100%;background:var(--accent);width:0;transition:width .12s}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mute)}.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
input[type=number],input[type=text],input[type=password]{font:inherit;padding:.3rem .5rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--ink);width:9rem}
button.act{font:inherit;padding:.4rem .8rem;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);cursor:pointer}
button.act.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.help{color:var(--mute);font-size:.84rem}.q{display:inline-block;width:1.1em;height:1.1em;border-radius:50%;border:1px solid var(--mute);color:var(--mute);font-size:.7em;text-align:center;line-height:1.1em;margin-left:.3em;cursor:help}
.log{font-family:"IBM Plex Mono",Consolas,monospace;font-size:.82rem;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;max-height:20rem;overflow:auto;white-space:pre-wrap}
.row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;margin:.5rem 0}code{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:.05em .35em;font-size:.9em}
h2{font-size:1.05rem;margin:1.2rem 0 .6rem}#toast{position:fixed;bottom:1rem;right:1rem;background:var(--ink);color:var(--bg);padding:.5rem .9rem;border-radius:6px;opacity:0;transition:opacity .2s}
img.snap{max-width:100%;border-radius:8px;border:1px solid var(--line)}
</style></head><body>
<header><h1>Talker control</h1><span class="sub" id="who">connecting…</span><span class="sub" id="up"></span></header>
<nav><button class="on" data-t="status">Status</button><button data-t="tune">Tune</button><button data-t="test">Test setup</button><button data-t="ref">Reference</button><button data-t="wifi">Wi-Fi</button></nav>
<main>
<section id="status" class="on">
  <div class="grid" id="cards"></div>
  <h2>Last turn</h2><div class="log" id="turn">…</div>
  <h2>Recent events</h2><div class="log" id="events"></div>
</section>
<section id="tune">
  <p class="help">Changes apply at once and last until the program exits. The flag next to each one sets the same thing at startup.</p>
  <div class="wrap"><table id="tunables"><tr><th>setting</th><th>value</th><th>what it does</th></tr></table></div>
</section>
<section id="test">
  <div class="grid">
    <div class="card"><h3>Mic level</h3><div class="big" id="lvl">0</div><div class="meter"><i id="lvlbar"></i></div><div class="help">Aim for 500 to 5,000 while someone talks from the visitor spot. Above 28,000 clips.</div></div>
    <div class="card"><h3>Speaking</h3><div class="big" id="spk">–</div><div class="help">What the speaker is doing right now.</div></div>
  </div>
  <h2>Try things</h2>
  <div id="actions"></div>
  <h2>Camera</h2>
  <div class="row"><button class="act" onclick="snap()">Take a snapshot</button><span class="help" id="camhelp"></span></div>
  <div id="snapbox"></div>
  <h2>Calibrate this room</h2>
  <p class="help">Measures the room, plays the character through the speaker and measures the bleed into the mic, then asks you to talk from the visitor spot. The loop is paused meanwhile. Result is saved to calibration.json and used as the default next start.</p>
  <div class="row"><button class="act primary" onclick="calib('start')">Start calibration</button><button class="act" id="cont" onclick="calib('continue')" disabled>I'm at the visitor spot, continue</button><span id="calmsg" class="help"></span></div>
  <div class="log" id="calres" style="display:none"></div>
</section>
<section id="ref">
  <h2>Command line flags <span class="help">(value in use this run, then the default)</span></h2>
  <div class="wrap"><table id="flags"><tr><th>flag</th><th>in use</th><th>default</th><th>help</th></tr></table></div>
  <h2>face.json fields for this face</h2>
  <div class="wrap"><table id="face"><tr><th>field</th><th>value</th><th>what it does</th></tr></table></div>
</section>
<section id="wifi">
  <p class="help">Through NetworkManager (nmcli), as on Raspberry Pi OS. Joining a network here keeps this page reachable only if the box stays on a network you can also reach.</p>
  <div id="wifistat" class="help"></div>
  <div class="row"><button class="act" onclick="wifi(true)">Scan for networks</button></div>
  <div class="wrap"><table id="nets"><tr><th>network</th><th>signal</th><th>security</th><th></th></tr></table></div>
  <div class="row"><input type="text" id="ssid" placeholder="network name"><input type="password" id="pw" placeholder="password"><button class="act primary" onclick="join()">Join</button><span id="wifimsg" class="help"></span></div>
</section>
</main>
<div id="toast"></div>
<script>
const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t));if(b.dataset.t==='ref')loadRef();if(b.dataset.t==='wifi')wifi(false);});
function toast(m){const t=$('#toast');t.textContent=m;t.style.opacity=1;clearTimeout(t._h);t._h=setTimeout(()=>t.style.opacity=0,2200)}
async function post(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json()}
let built=false;
function buildTunables(list){const t=$('#tunables');list.forEach(x=>{const tr=document.createElement('tr');const ctl=x.kind==='bool'?`<input type="checkbox" data-n="${x.name}" ${x.value?'checked':''}>`:`<input type="${x.kind==='str'?'text':'number'}" data-n="${x.name}" value="${esc(x.value)}" ${x.lo!=null?`min="${x.lo}"`:''} ${x.hi!=null?`max="${x.hi}"`:''} step="${x.kind==='int'?1:'any'}"> <span class="mute">${esc(x.unit)}</span>`;tr.innerHTML=`<td><b>${esc(x.name)}</b>${x.flag?`<div class="help"><code>${esc(x.flag)}</code></div>`:''}</td><td>${ctl}</td><td class="help">${esc(x.help)}</td>`;t.appendChild(tr);});
 t.querySelectorAll('input').forEach(i=>i.onchange=async()=>{const v=i.type==='checkbox'?i.checked:i.value;const r=await post('/api/set',{name:i.dataset.n,value:v});toast(r.error||`${i.dataset.n} = ${r.value}`)});built=true}
function buildActions(list){const d=$('#actions');d.innerHTML='';list.forEach(a=>{const row=document.createElement('div');row.className='row';row.innerHTML=`<button class="act">${esc(a.name)}</button>${a.takes_text?`<input type="text" style="width:22rem" placeholder="text">`:''}<span class="help">${esc(a.help)}</span>`;row.querySelector('button').onclick=async()=>{const inp=row.querySelector('input');const r=await post('/api/action',{name:a.name,text:inp?inp.value:''});toast(r.error||r.message)};d.appendChild(row)})}
function card(h,v,cls,help){return `<div class="card"><h3>${esc(h)}</h3><div class="big ${cls||''}">${esc(v)}</div>${help?`<div class="help">${esc(help)}</div>`:''}</div>`}
async function tick(){try{const s=await (await fetch('/api/state')).json();const st=s.status||{};$('#who').textContent=`${st.face||'?'} · ears ${st.stt||'-'} · brain ${st.llm||'-'} · voice ${st.tts||'-'}`;$('#up').textContent=`up ${Math.floor(s.uptime_s/60)} min`;
 const mode=st.waiting?'waiting':(st.engaged===false?'dormant':(st.thinking?'thinking':(st.speaking?'speaking':'listening')));
 $('#cards').innerHTML=card('State',mode,mode==='speaking'?'ok':(mode==='dormant'?'warn':''),st.mode_help||'')+card('Mic level',Math.round(st.mic_rms||0),'', 'live, from the microphone')+card('Time to first audio',st.first_audio_ms!=null?st.first_audio_ms+' ms':'–','','last reply, from the moment the text was ready')+card('Heard',st.last_heard||'–','','')+card('Said',st.last_said||'–','','')+(st.vision?card('Vision',`${st.vision.described} looks, ${st.vision.skipped} skipped`,'',st.vision.latest||''):'');
 $('#turn').textContent=st.last_turn||'no turn yet';
 $('#events').innerHTML=(s.events||[]).slice().reverse().map(e=>`<div><span class="mute">${new Date(e.t*1000).toLocaleTimeString()}</span> [${esc(e.kind)}] ${esc(e.text)}</div>`).join('');
 $('#lvl').textContent=Math.round(st.mic_rms||0);$('#lvlbar').style.width=Math.min(100,Math.log10(1+(st.mic_rms||0))/4.5*100)+'%';$('#spk').textContent=st.speaking?'speaking':'quiet';
 if(!built){buildTunables(s.tunables);buildActions(s.actions);$('#camhelp').textContent=s.has_camera?'':'no camera in this run (start with --camera)';}
 const c=s.calibration||{};$('#cont').disabled=c.state!=='waiting';$('#calmsg').textContent=c.state==='idle'?'':`${c.state}: ${c.prompt||c.error||''}`;if(c.result){const r=c.result;$('#calres').style.display='block';$('#calres').textContent=`ambient ${r.ambient}  speaker bleed ${r.speaker}  person ${r.person}  (person is ${r.person_over_speaker}x the speaker)\n${r.verdict}\nrecommended --barge-in-boost ${r.barge_in_boost}${r.mic_gain!=='ok'?'\nmic gain: '+r.mic_gain:''}`}
}catch(e){$('#who').textContent='not reachable: '+e}}
async function loadRef(){const r=await (await fetch('/api/reference')).json();$('#flags').innerHTML='<tr><th>flag</th><th>in use</th><th>default</th><th>help</th></tr>'+r.flags.map(f=>`<tr><td><code>${esc(f.flags)}</code></td><td>${esc(f.value)}</td><td class="mute">${esc(f.default)}</td><td class="help">${esc(f.help)}</td></tr>`).join('');$('#face').innerHTML='<tr><th>field</th><th>value</th><th>what it does</th></tr>'+r.face.map(f=>`<tr><td><code>${esc(f.field)}</code></td><td><code>${esc(f.value)}</code></td><td class="help">${esc(f.help)}</td></tr>`).join('')}
async function snap(){const b=$('#snapbox');b.innerHTML='<span class="help">taking…</span>';const r=await fetch('/api/snapshot.jpg?'+Date.now());if(!r.ok){b.innerHTML='<span class="help">'+esc((await r.json()).error)+'</span>';return}const u=URL.createObjectURL(await r.blob());b.innerHTML=`<img class="snap" src="${u}">`}
async function calib(step){const r=await post('/api/calibrate',{step});if(r.error)toast(r.error)}
async function wifi(scan){const r=await (await fetch('/api/wifi'+(scan?'?scan=1':''))).json();if(!r.available){$('#wifistat').textContent=r.reason;return}$('#wifistat').innerHTML=(r.devices||[]).map(d=>`${esc(d.device)} (${esc(d.type)}): <b>${esc(d.state)}</b> ${esc(d.connection)}`).join(' · ')+` · this box is ${esc(r.ip)}`;if(r.networks){$('#nets').innerHTML='<tr><th>network</th><th>signal</th><th>security</th><th></th></tr>'+r.networks.map(n=>`<tr><td>${n.active?'<b>':''}${esc(n.ssid)}${n.active?'</b> (connected)':''}</td><td>${n.signal}%</td><td>${esc(n.security)}</td><td><button class="act" onclick="$('#ssid').value=${JSON.stringify(n.ssid)}">use</button></td></tr>`).join('')}}
async function join(){$('#wifimsg').textContent='joining…';const r=await post('/api/wifi/connect',{ssid:$('#ssid').value,password:$('#pw').value});$('#wifimsg').textContent=r.error||r.message;wifi(false)}
tick();setInterval(tick,500);
</script></body></html>
"""
