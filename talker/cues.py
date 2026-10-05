"""
cues.py — short spoken lines that still play when the cloud is gone.

The character's voice is a cloud service, so "I've lost my internet connection"
cannot be synthesized at the moment it is needed. prepare() renders each line
once while online (edge-tts, then ffmpeg to WAV) into ~/.cache/talker/cues/ and
play() plays the cached file locally (pw-play, else aplay). A cue that was never
rendered is skipped silently.

    cues.prepare({"offline": "I've lost my internet connection."})   # at startup, in the background
    cues.play("offline")                                             # later, any thread
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import threading
from typing import Dict, Optional

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "talker", "cues")
VOICE = "en-US-AriaNeural"
_paths: Dict[str, str] = {}


def _path(key: str, text: str) -> str:
    digest = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:10]
    return os.path.join(CACHE, f"{key}-{digest}.wav")


def _render(text: str, wav: str) -> bool:
    try:
        import edge_tts
    except ImportError:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    mp3 = wav[:-4] + ".mp3"

    async def fetch():
        await edge_tts.Communicate(text, VOICE).save(mp3)
    try:
        asyncio.run(fetch())
        subprocess.run([ffmpeg, "-loglevel", "quiet", "-y", "-i", mp3, wav], check=True, timeout=60)
        return os.path.isfile(wav)
    except Exception:
        return False
    finally:
        try:
            os.remove(mp3)
        except OSError:
            pass


def prepare(lines: Dict[str, str], background: bool = True) -> None:
    def work():
        os.makedirs(CACHE, exist_ok=True)
        for key, text in lines.items():
            wav = _path(key, text)
            if os.path.isfile(wav) or _render(text, wav):
                _paths[key] = wav
            else:
                print(f"[cues] could not prepare '{key}' (offline at startup?); it will be skipped")
    if background:
        threading.Thread(target=work, name="cues", daemon=True).start()
    else:
        work()


def play(key: str) -> bool:
    wav: Optional[str] = _paths.get(key)
    if not wav or not os.path.isfile(wav):
        return False
    player = shutil.which("pw-play") or shutil.which("aplay")
    if not player:
        return False
    subprocess.Popen([player, wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True
