#!/usr/bin/env python3
"""
bench_tts_local.py — compare local (offline) text-to-speech engines for Talker.

Speed vs quality, CPU only, with a fixed thread budget so the numbers say
something about a Raspberry Pi 5 (4 cores). Every engine gets the same
sentences; each runs in its own subprocess so memory and threads are isolated.
Measured per sentence:

    ttfa   time to first audio (ms): what the listener waits. Streaming engines
           report their first chunk; the rest report the whole synthesis.
    synth  wall time to synthesize the whole sentence (ms)
    audio  seconds of speech produced
    rtf    synth / audio: below 1.0 keeps up with itself (a rough Pi 5 estimate
           is 3-4x the desktop figure with the same 4 threads)

WAVs go to --out so you can listen to the quality side by side.

    .bench-venv/bin/python tools/bench_tts_local.py                  # everything installed
    .bench-venv/bin/python tools/bench_tts_local.py --engines piper,kokoro --threads 4

Engines (skipped with a note when not installed): espeak, flite, piper, kokoro,
melo, vits, tacotron2, xtts, chattts. See README, Local voices.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("TTS_BENCH_MODELS", HERE.parent / ".bench-models"))

SENTENCES = [
    ("short", "Welcome in! Mind the step."),
    ("medium", "Oh, hello there. I was starting to think you had gotten lost on the way up the hill. "
               "Come closer, I don't bite. Much."),
    ("long", "The weather tonight is partly cloudy with a chance of bats, so keep your candles lit and "
             "your candy close. If you hear creaking, that is probably just the porch. Probably. "
             "Now, what brings a brave soul like you to my doorstep?"),
]

ENGINE_NAMES = ["espeak", "flite", "piper", "kokoro", "melo", "vits", "tacotron2", "xtts", "chattts"]

# ── engines ──────────────────────────────────────────────────────────────────
# Each returns (sample_rate, synth(text) -> iterator of int16 numpy chunks).
# Chunked output is what makes "time to first audio" meaningful.


def _cli_engine(cmd: Callable[[str, str], List[str]]):
    def synth(text: str) -> Iterator[Tuple[int, np.ndarray]]:
        out = "/tmp/tts_bench_cli.wav"
        subprocess.run(cmd(text, out), check=True, capture_output=True)
        with wave.open(out) as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        yield sr, data
    return synth


def engine_espeak():
    subprocess.run(["espeak-ng", "--version"], check=True, capture_output=True)
    return _cli_engine(lambda t, o: ["espeak-ng", "-v", "en-us", "-s", "165", "-w", o, t])


def engine_flite():
    subprocess.run(["flite", "--version"], capture_output=True)   # exits 1 even when fine
    return _cli_engine(lambda t, o: ["flite", "-voice", "slt", "-t", t, "-o", o])


def _ort_session(path: str, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def engine_piper(threads: int):
    from piper import PiperVoice
    model = next(MODELS.glob("piper/*.onnx"), None)
    if model is None:
        raise RuntimeError(f"no Piper voice in {MODELS / 'piper'} "
                           "(python -m piper.download_voices en_US-lessac-medium --data-dir .bench-models/piper)")
    voice = PiperVoice.load(str(model))
    voice.session = _ort_session(str(model), threads)

    def synth(text: str):
        for chunk in voice.synthesize(text):          # piper >= 1.3 streams AudioChunk per sentence
            pcm = getattr(chunk, "audio_int16_bytes", None)
            if pcm is None:                           # older API: raw bytes
                pcm = chunk
            yield chunk.sample_rate if hasattr(chunk, "sample_rate") else voice.config.sample_rate, \
                np.frombuffer(pcm, dtype=np.int16)
    return synth


def engine_kokoro(threads: int):
    from kokoro_onnx import Kokoro
    model, voices = MODELS / "kokoro/kokoro-v1.0.onnx", MODELS / "kokoro/voices-v1.0.bin"
    if not model.exists():
        raise RuntimeError(f"no Kokoro model in {MODELS / 'kokoro'}")
    k = Kokoro.from_session(_ort_session(str(model), threads), str(voices))

    def synth(text: str):
        async def run(q: asyncio.Queue):
            async for samples, sr in k.create_stream(text, voice="af_heart", speed=1.0, lang="en-us"):
                q.put_nowait((sr, (np.asarray(samples) * 32767).astype(np.int16)))
            q.put_nowait(None)
        # drive the async stream synchronously, yielding chunks as they land
        loop = asyncio.new_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        task = loop.create_task(run(q))
        try:
            while True:
                while q.empty() and not task.done():
                    loop.run_until_complete(asyncio.sleep(0.005))
                if q.empty() and task.done():
                    task.result()
                    break
                item = q.get_nowait()
                if item is None:
                    break
                yield item
        finally:
            loop.close()
    return synth


def _coqui(model_name: str, threads: int, **kw):
    os.environ["COQUI_TOS_AGREED"] = "1"
    import torch
    torch.set_num_threads(threads)
    from TTS.api import TTS
    tts = TTS(model_name, progress_bar=False).to("cpu")
    sr = tts.synthesizer.output_sample_rate

    def synth(text: str):
        wav = tts.tts(text, **kw)
        yield sr, (np.asarray(wav) * 32767).astype(np.int16)
    return synth


def engine_vits(threads: int):
    return _coqui("tts_models/en/ljspeech/vits", threads)


def engine_tacotron2(threads: int):
    return _coqui("tts_models/en/ljspeech/tacotron2-DDC", threads)


def engine_xtts(threads: int):
    return _coqui("tts_models/multilingual/multi-dataset/xtts_v2", threads,
                  speaker="Ana Florence", language="en")


def engine_chattts(threads: int):
    import torch
    torch.set_num_threads(threads)
    import ChatTTS
    chat = ChatTTS.Chat()
    if not chat.load(compile=False, source="huggingface"):
        raise RuntimeError("ChatTTS model download failed")
    spk = chat.sample_random_speaker()

    def synth(text: str):
        params = ChatTTS.Chat.InferCodeParams(spk_emb=spk, temperature=0.3)
        wavs = chat.infer([text], params_infer_code=params, skip_refine_text=True)
        yield 24000, (np.asarray(wavs[0]).reshape(-1) * 32767).astype(np.int16)
    return synth


def engine_melo(threads: int):
    import torch
    torch.set_num_threads(threads)
    from melo.api import TTS as MeloTTS
    m = MeloTTS(language="EN", device="cpu")
    spk = m.hps.data.spk2id["EN-US"]

    def synth(text: str):
        wav = m.tts_to_file(text, spk, None, speed=1.0, quiet=True)
        yield m.hps.data.sampling_rate, (np.asarray(wav) * 32767).astype(np.int16)
    return synth


def build(name: str, threads: int):
    if name == "espeak":
        return engine_espeak()
    if name == "flite":
        return engine_flite()
    return globals()[f"engine_{name}"](threads)


# ── one engine, in its own process ──────────────────────────────────────────

def run_engine(name: str, threads: int, out: Path) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    t0 = time.perf_counter()
    synth = build(name, threads)
    load_s = time.perf_counter() - t0
    rows = []
    for label, text in SENTENCES:
        if label == "short":                          # warm-up: first call pays one-off costs
            for _ in synth(text):
                pass
        t_start = time.perf_counter()
        first = None
        chunks: List[np.ndarray] = []
        sr = 0
        for sr, pcm in synth(text):
            if first is None and len(pcm):
                first = time.perf_counter() - t_start
            chunks.append(pcm)
        total = time.perf_counter() - t_start
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
        secs = len(audio) / sr if sr else 0.0
        path = out / f"{name}_{label}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
        rows.append({"sentence": label, "ttfa_ms": round((first or total) * 1000),
                     "synth_ms": round(total * 1000), "audio_s": round(secs, 2),
                     "rtf": round(total / secs, 2) if secs else None, "sample_rate": sr})
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"engine": name, "load_s": round(load_s, 1), "peak_rss_mb": round(rss_mb), "rows": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines", default=",".join(ENGINE_NAMES))
    p.add_argument("--threads", type=int, default=4, help="CPU threads per engine (Pi 5 has 4 cores)")
    p.add_argument("--out", default=str(HERE.parent / "logs" / "tts_bench"))
    p.add_argument("--_engine", help=argparse.SUPPRESS)   # child mode
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args._engine:
        try:
            print(json.dumps(run_engine(args._engine, args.threads, out)))
        except Exception as e:
            print(json.dumps({"engine": args._engine, "error": f"{type(e).__name__}: {e}"}))
        return 0

    results = []
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        print(f"[bench] {name} ...", file=sys.stderr, flush=True)
        proc = subprocess.run([sys.executable, __file__, "--_engine", name, "--threads", str(args.threads),
                               "--out", str(out)], capture_output=True, text=True, timeout=3600)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        res = json.loads(line[-1]) if line else {"engine": name, "error": (proc.stderr or "no output")[-400:]}
        if "error" in res:
            print(f"[bench] {name}: {res['error'].splitlines()[-1]}", file=sys.stderr)
        results.append(res)

    # merge with earlier runs of other engines (results.json accumulates)
    prev = {}
    if (out / "results.json").exists():
        try:
            prev = {r["engine"]: r for r in json.loads((out / "results.json").read_text()).get("results", [])}
        except ValueError:
            prev = {}
    for r in results:
        prev[r["engine"]] = r
    results = [prev[n] for n in ENGINE_NAMES if n in prev] + [r for n, r in prev.items() if n not in ENGINE_NAMES]
    (out / "results.json").write_text(json.dumps({"threads": args.threads, "results": results}, indent=1))
    print(f"\nLocal TTS, {args.threads} CPU threads each. ttfa = time to first audio, rtf = synth time / audio time\n")
    print("| engine | load s | RAM MB | short ttfa | medium ttfa | long ttfa | medium rtf | long rtf | rate |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if "error" in r:
            print(f"| {r['engine']} | not run | | | | | | | |")
            continue
        by = {row["sentence"]: row for row in r["rows"]}
        print(f"| {r['engine']} | {r['load_s']} | {r['peak_rss_mb']} | {by['short']['ttfa_ms']} ms | "
              f"{by['medium']['ttfa_ms']} ms | {by['long']['ttfa_ms']} ms | {by['medium']['rtf']} | "
              f"{by['long']['rtf']} | {by['long']['sample_rate']} |")
    print(f"\nWAVs in {out}/  (listen: <engine>_<short|medium|long>.wav)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
