#!/usr/bin/env python3
"""
tools/bench_e2e.py — question in, voice out: the brain + voice chain end to end.

Runs the same questions through real brain/voice pairs and reports, per pair,
what a listener would wait, measured from the moment the question is handed to
the brain (the same instant the microphone would hand over a transcript):

    first token   brain starts answering
    first audio   first PCM leaves the voice: what the listener actually waits
    reply done    the whole reply is synthesized
    speech s      seconds of speech produced

Pairs (--pairs, comma separated):
    cloud              claude-haiku-4-5 + ElevenLabs flash      (keys needed)
    local              Ollama + Piper                           (offline; OLLAMA_MODEL or --ollama-model)
    claude+piper       cloud brain, local voice
    ollama+elevenlabs  local brain, cloud voice

    python tools/bench_e2e.py --pairs cloud,local --ollama-model <name>

Speech-to-text is not included: it is the same local Whisper in every pair.
"""
from __future__ import annotations

import os as _os
import sys as _sys

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

import argparse
import statistics
import time
from typing import Dict, List, Optional

_os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from talker.env_config import load_dotenv

QUESTIONS = [
    "Who goes there?",
    "What's your favorite thing about Halloween?",
    "Can we come in?",
    "What's two plus two?",
]
CHARACTER = "a goofy jack-o-lantern on a porch, fond of visitors"

PAIRS = {
    "cloud": ("claude", "elevenlabs"),
    "local": ("ollama", "piper"),
    "claude+piper": ("claude", "piper"),
    "ollama+elevenlabs": ("ollama", "elevenlabs"),
}


def make_brain(kind: str, ollama_model: Optional[str]):
    if kind == "claude":
        from talker.brains.claude_chat import ClaudeChat
        return ClaudeChat(model="claude-haiku-4-5", thinking=False, character=CHARACTER)
    from talker.brains.ollama_chat import OllamaChat
    chat = OllamaChat(model=ollama_model, character=CHARACTER)
    chat.warm_up()
    return chat


def make_voice(kind: str):
    from talker.tts_backends import make_backend
    if kind == "elevenlabs":
        return make_backend("elevenlabs", model="flash")
    return make_backend(kind)


def wait_until(pred, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


def run_pair(name: str, brain_kind: str, voice_kind: str, questions: List[str],
             ollama_model: Optional[str]) -> List[Dict]:
    from talker.audio_engine import NullAudioEngine
    from talker.phoneme_scheduler import ScheduleReader
    from talker.speech_pipeline import SpeechPipeline
    chat = make_brain(brain_kind, ollama_model)
    voice = make_voice(voice_kind)
    pipeline = SpeechPipeline(NullAudioEngine(sample_rate=voice.sample_rate), ScheduleReader(), voice,
                              lead_seconds=0.0)
    pipeline.start()
    rows = []
    try:
        for q in questions:
            pipeline.stats = {}
            pipeline.first_audio_at = 0.0
            marks: Dict[str, float] = {}
            t0 = time.monotonic()

            def chunks():
                for c in chat.reply(q):
                    marks.setdefault("first_token", time.monotonic())
                    yield c
                marks["last_token"] = time.monotonic()

            pipeline.speak_stream(chunks())
            ok = wait_until(lambda: bool(pipeline.stats.get("words")), timeout=60)
            t_done = time.monotonic()
            pipeline.interrupt()                       # drop the simulated playback before the next one
            row = {
                "pair": name, "question": q, "ok": ok,
                "first_token_ms": round((marks.get("first_token", t_done) - t0) * 1000),
                "first_audio_ms": round((pipeline.first_audio_at - t0) * 1000) if pipeline.first_audio_at else None,
                "reply_done_ms": round((t_done - t0) * 1000),
                "speech_s": pipeline.stats.get("audio_seconds"),
                "words": pipeline.stats.get("words"),
            }
            rows.append(row)
            print(f"  {name:18s} {row['first_token_ms']:5d} ms token  {row['first_audio_ms']!s:>5} ms audio  "
                  f"{row['reply_done_ms']:5d} ms done  {row['speech_s']} s  «{q}»", flush=True)
            time.sleep(0.3)
    finally:
        pipeline.stop()
        close = getattr(voice, "close", None)
        if close is not None:
            try:
                import asyncio
                asyncio.run(close())
            except Exception:
                pass
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default="cloud,local")
    p.add_argument("--ollama-model", default=None, help="model for the local brain (or OLLAMA_MODEL)")
    p.add_argument("--questions", type=int, default=len(QUESTIONS))
    args = p.parse_args()
    load_dotenv()
    ollama_model = args.ollama_model or _os.environ.get("OLLAMA_MODEL")
    questions = QUESTIONS[:args.questions]

    results: Dict[str, List[Dict]] = {}
    for name in [x.strip() for x in args.pairs.split(",") if x.strip()]:
        brain, voice = PAIRS[name]
        print(f"[e2e] {name}: {brain} + {voice}")
        try:
            results[name] = run_pair(name, brain, voice, questions, ollama_model)
        except Exception as e:
            print(f"[e2e] {name} failed: {type(e).__name__}: {e}")

    def med(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(statistics.median(vals)) if vals else "-"

    print("\nQuestion in -> voice out, median over the questions (ms). STT not included, same in every pair.\n")
    print("| pair | brain | voice | first token | first audio | reply done | speech s |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name, rows in results.items():
        b, v = PAIRS[name]
        sp = [r["speech_s"] for r in rows if r.get("speech_s")]
        print(f"| {name} | {b} | {v} | {med(rows, 'first_token_ms')} | {med(rows, 'first_audio_ms')} | "
              f"{med(rows, 'reply_done_ms')} | {round(statistics.median(sp), 1) if sp else '-'} |")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
