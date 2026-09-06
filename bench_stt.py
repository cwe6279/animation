"""
bench_stt.py — accuracy and speed of every speech-to-text backend on the same clips.

    python bench_stt.py --synth                # edge-tts clips, clean + noisy (no mic needed)
    python bench_stt.py recordings/            # your own clips from: voice_loop.py --mic-test --record recordings/
    python bench_stt.py recordings/ --backends whisper:base.en,whisper:small.en,groq,elevenlabs

A recordings folder holds 16 kHz mono WAVs and a transcripts.txt with
"<file>\\t<reference text>" lines (the mic test writes a draft; correct it).
Reports word error rate (lower is better) and seconds per clip.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import subprocess
import sys
import time
import wave
from typing import Callable, Dict, List, Tuple

import numpy as np

from env_config import load_dotenv

SENTENCES = [
    "What is the weather like on Mars today, and should I bring a jacket?",
    "Tell me a spooky story about a pumpkin that learned to talk.",
    "My favorite ice cream flavor is mint chocolate chip, what about you?",
    "Can you turn the lights down a little, it's too bright in here.",
    "Who goes there? Speak now or forever hold your peace.",
    "I'll be back in fifteen minutes, keep an eye on the door.",
]
VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural"]


# ─── clips ───────────────────────────────────────────────────────────
def synth_pcm(text: str, voice: str) -> bytes:
    import edge_tts
    from tts_backends import find_ffmpeg

    async def run():
        mp3 = b""
        async for ch in edge_tts.Communicate(text, voice=voice).stream():
            if ch["type"] == "audio":
                mp3 += ch["data"]
        return mp3
    mp3 = asyncio.run(run())
    return subprocess.run([find_ffmpeg(), "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                           "-ar", "16000", "-ac", "1", "pipe:1"], input=mp3, capture_output=True).stdout


def add_noise(pcm: bytes, snr_db: float, seed: int = 0) -> bytes:
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(s * s)) or 1.0
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.default_rng(seed).standard_normal(s.size) * noise_rms
    return np.clip(s + noise, -32768, 32767).astype(np.int16).tobytes()


def synth_clips() -> List[Tuple[str, bytes, str]]:
    clips = []
    for i, text in enumerate(SENTENCES):
        voice = VOICES[i % len(VOICES)]
        print(f"  synthesizing {i + 1}/{len(SENTENCES)} ({voice})", end="\r")
        pcm = synth_pcm(text, voice)
        clips.append((f"clean-{i + 1}", pcm, text))
        clips.append((f"noisy10dB-{i + 1}", add_noise(pcm, 10, i), text))
    print()
    return clips


def load_clips(folder: str) -> List[Tuple[str, bytes, str]]:
    refs: Dict[str, str] = {}
    with open(os.path.join(folder, "transcripts.txt"), encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                name, ref = line.rstrip("\n").split("\t", 1)
                refs[name.strip()] = ref.strip()
    clips = []
    for name, ref in refs.items():
        with wave.open(os.path.join(folder, name), "rb") as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, name
            clips.append((name, w.readframes(w.getnframes()), ref))
    return clips


# ─── metric ──────────────────────────────────────────────────────────
def norm_words(text: str) -> List[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def wer(ref: str, hyp: str) -> Tuple[float, int]:
    r, h = norm_words(ref), norm_words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return (d[len(r), len(h)] / max(1, len(r))), len(r)


# ─── backends as plain "pcm -> text" functions ───────────────────────
def make_transcribers(specs: List[str]) -> Dict[str, Callable[[bytes], str]]:
    out: Dict[str, Callable[[bytes], str]] = {}
    for spec in specs:
        name, _, opt = spec.partition(":")
        try:
            if name == "whisper":
                from faster_whisper import WhisperModel
                from stt_backends import CACHE_DIR
                size = opt or "base.en"
                model = WhisperModel(size, device="cpu", compute_type="int8",
                                     download_root=os.path.join(CACHE_DIR, "whisper"))

                def f(pcm, model=model):
                    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    segs, _ = model.transcribe(s, language="en", beam_size=1, condition_on_previous_text=False)
                    return " ".join(x.text.strip() for x in segs)
                out[f"whisper {size}"] = f
            elif name == "vosk":
                from stt_backends import VoskSTT
                v = VoskSTT(model_path=opt or None)

                def f(pcm, v=v):
                    v.reset()
                    text = ""
                    for i in range(0, len(pcm), 4000):
                        t = v.feed(pcm[i:i + 4000])
                        if t and t.final:
                            text += " " + t.text
                    import json
                    text += " " + json.loads(v._rec.FinalResult()).get("text", "")
                    return text.strip()
                out[f"vosk {opt or 'small'}"] = f
            elif name in ("groq", "openai"):
                from stt_backends import OpenAICompatSTT
                be = OpenAICompatSTT.groq(model=opt or None) if name == "groq" else OpenAICompatSTT.openai(model=opt or None)

                def f(pcm, be=be):
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
                    buf.seek(0); buf.name = "u.wav"
                    r = be._client.audio.transcriptions.create(model=be.model, file=buf, language="en", response_format="text")
                    return r if isinstance(r, str) else getattr(r, "text", "")
                out[f"{name} {be.model}"] = f
            elif name == "elevenlabs":
                key = os.environ.get("ELEVENLABS_API_KEY")
                if not key:
                    raise RuntimeError("ELEVENLABS_API_KEY not set")

                def f(pcm, key=key, model=opt or "scribe_v1"):
                    import aiohttp

                    async def run():
                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as w:
                            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
                        form = aiohttp.FormData()
                        form.add_field("model_id", model)
                        form.add_field("file", buf.getvalue(), filename="u.wav", content_type="audio/wav")
                        async with aiohttp.ClientSession() as http:
                            async with http.post("https://api.elevenlabs.io/v1/speech-to-text",
                                                 headers={"xi-api-key": key}, data=form) as r:
                                d = await r.json()
                                return d.get("text") or f"ERROR {str(d)[:80]}"
                    return asyncio.run(run())
                out[f"elevenlabs {opt or 'scribe_v1'}"] = f
            else:
                print(f"  unknown backend {spec!r}")
        except ImportError as e:
            print(f"  skipping {spec}: package not installed ({e})")
        except Exception as e:
            print(f"  skipping {spec}: {str(e)[:120]}")
    return out


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="recordings folder (WAVs + transcripts.txt)")
    p.add_argument("--synth", action="store_true", help="use synthesized clips instead of a folder")
    p.add_argument("--backends", default="whisper:base.en,whisper:small.en,vosk,groq,elevenlabs",
                   help="comma list: whisper[:size], vosk[:model], groq[:model], openai[:model], elevenlabs[:model]")
    args = p.parse_args()
    if not args.synth and not args.folder:
        p.error("give a recordings folder or --synth")

    clips = synth_clips() if args.synth else load_clips(args.folder)
    print(f"{len(clips)} clips")
    transcribers = make_transcribers(args.backends.split(","))

    print(f"\n{'backend':28s} {'WER':>6s}  {'clean':>6s}  {'noisy':>6s}  {'s/clip':>6s}")
    for label, fn in transcribers.items():
        errs = words = 0
        buckets: Dict[str, List[float]] = {"clean": [], "noisy": []}
        times = []
        worst = ("", 0.0, "")
        for name, pcm, ref in clips:
            t0 = time.monotonic()
            try:
                hyp = fn(pcm)
            except Exception as e:
                hyp = f"ERROR {e}"
            times.append(time.monotonic() - t0)
            w, n = wer(ref, hyp)
            errs += w * n
            words += n
            buckets["noisy" if name.startswith("noisy") else "clean"].append(w)
            if w > worst[1]:
                worst = (name, w, hyp)
        clean = np.mean(buckets["clean"]) if buckets["clean"] else float("nan")
        noisy = np.mean(buckets["noisy"]) if buckets["noisy"] else float("nan")
        print(f"{label:28s} {errs / max(1, words):6.1%}  {clean:6.1%}  {noisy:6.1%}  {np.mean(times):6.2f}")
        if worst[1] > 0:
            print(f"{'':28s} worst: {worst[0]} ({worst[1]:.0%}): {worst[2].strip()[:90]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
