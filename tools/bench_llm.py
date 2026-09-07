"""
tools/bench_llm.py — how well each brain follows the performance-tag spec, plus speed.

    python tools/bench_llm.py                       # all configured brains
    python tools/bench_llm.py --llms "claude:claude-haiku-4-5,openai:gpt-4o-mini"

Scores per reply (against talker/brains/system_prompt.md):
  known tags    share of tags that are in the allowed vocabulary / face map
  leading       share of tags placed before words (not trailing a sentence)
  tags/sent     tags per sentence (the prompt asks for most sentences to have none)
  markdown      replies containing markdown, lists or emoji
  words         average reply length
"""

from __future__ import annotations

import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

import argparse
import re
import statistics
import sys
import time
from typing import Dict, List

from talker.env_config import load_dotenv
from talker.phoneme_scheduler import _TAG_RE, tag_to_emotion

PROMPTS = [
    "Who goes there?",
    "What's your favorite ice cream?",
    "Tell me something scary about this house.",
    "I'm feeling a bit sad today.",
    "What's two plus two?",
    "Can you sing me a song?",
    "Do you like dogs?",
    "Goodbye, see you next year.",
]
CHAR = "a sly green Halloween cat: playful, a bit spooky, likes to tease"

# Vocabulary the prompt allows verbatim, on top of anything the face map understands.
PROMPT_VOCAB = {
    "excited", "nervous", "frustrated", "sorrowful", "calm", "happy", "angry", "sad", "annoyed",
    "surprised", "curious", "tired", "amazed", "scared", "sigh", "laughs", "giggle", "chuckle",
    "gasps", "gulps", "whispers", "sigh of relief", "light chuckle", "pauses", "hesitates",
    "stammers", "resigned tone", "cheerfully", "flatly", "deadpan", "playfully", "sarcastically",
    "dramatic", "matter-of-fact", "whiny", "british accent", "southern us accent", "pirate voice",
    "sci-fi ai voice", "classic film noir",
}
MD_RE = re.compile(r"(\*\*|__|^#|^\s*[-*]\s|^\s*\d+\.\s|`|[\U0001F300-\U0001FAFF☀-➿])", re.M)


def known(tag: str) -> bool:
    t = tag.strip().lower()
    return t in PROMPT_VOCAB or tag_to_emotion(t) is not None or t.endswith(" accent") or t.endswith(" voice")


def score(reply: str) -> Dict[str, float]:
    tags = [m.group(1) for m in _TAG_RE.finditer(reply)]
    text = _TAG_RE.sub(" ", reply)
    sentences = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    # a tag is "leading" if the next non-space char is a letter/quote (words follow it)
    leading = 0
    for m in _TAG_RE.finditer(reply):
        rest = reply[m.end():].lstrip()
        if rest and (rest[0].isalnum() or rest[0] in "\"'([“‘"):
            leading += 1
    return {
        "tags": len(tags),
        "known": (sum(known(t) for t in tags) / len(tags)) if tags else 1.0,
        "leading": (leading / len(tags)) if tags else 1.0,
        "tags_per_sent": len(tags) / sentences,
        "markdown": 1.0 if MD_RE.search(reply) else 0.0,
        "words": len(text.split()),
        "unknown": [t for t in tags if not known(t)],
    }


def make_chat(spec: str):
    kind, _, model = spec.partition(":")
    if kind == "claude":
        from talker.brains.claude_chat import ClaudeChat
        return ClaudeChat(model=model or "claude-opus-5", character=CHAR, thinking=False), spec
    from talker.brains.openai_compat_chat import OpenAICompatChat
    chat = OpenAICompatChat.openai(model=model or None, character=CHAR)
    return chat, f"{kind}:{chat.model}"


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--llms", default="claude:claude-opus-5,claude:claude-haiku-4-5,openai:gpt-4o-mini")
    p.add_argument("--samples", type=int, default=2, help="replies to print per model")
    args = p.parse_args()

    rows = []
    samples: Dict[str, List[str]] = {}
    for spec in args.llms.split(","):
        try:
            chat, label = make_chat(spec.strip())
        except Exception as e:
            print(f"skip {spec}: {str(e)[:100]}")
            continue
        scores, firsts = [], []
        samples[label] = []
        for q in PROMPTS:
            chat.messages = []                       # independent single-turn replies
            t0 = time.monotonic(); first = None; parts = []
            try:
                for c in chat.reply(q):
                    if first is None:
                        first = time.monotonic() - t0
                    parts.append(c)
            except Exception as e:
                print(f"  {label}: ERROR {str(e)[:100]}")
                break
            reply = "".join(parts).strip()
            scores.append(score(reply))
            firsts.append((first or 0) * 1000)
            samples[label].append(f"Q: {q}\n   A: {reply}")
        if not scores:
            continue
        unknown = sorted({t for s in scores for t in s["unknown"]})
        rows.append((label, statistics.median(firsts), statistics.mean(s["known"] for s in scores),
                     statistics.mean(s["leading"] for s in scores), statistics.mean(s["tags_per_sent"] for s in scores),
                     statistics.mean(s["markdown"] for s in scores), statistics.mean(s["words"] for s in scores),
                     statistics.mean(s["tags"] for s in scores), unknown))

    print(f"\n{'model':32s} {'1st tok':>7s} {'known':>6s} {'leading':>7s} {'tags/sent':>9s} {'md':>4s} {'words':>5s} {'tags':>4s}")
    for label, ft, kn, ld, tps, md, words, tags, unknown in rows:
        print(f"{label:32s} {ft:6.0f}ms {kn:6.0%} {ld:7.0%} {tps:9.2f} {md:4.0%} {words:5.0f} {tags:4.1f}")
        if unknown:
            print(f"{'':32s} unknown tags: {', '.join(unknown[:8])}")
    for label, reps in samples.items():
        print(f"\n== {label}")
        for r in reps[:args.samples]:
            print("  " + r.replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
