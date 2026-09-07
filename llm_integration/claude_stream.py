"""
claude_stream.py — stream a Claude reply into the talking face.

Prints the model's text to stdout token by token, so it can be piped straight
into talker.py, which starts speaking the first sentence while the rest is
still being generated:

    python llm_integration/claude_stream.py "Tell me a spooky story" \\
        | python talker.py --face skull --stdin

Reads the emotion-tag instructions from system_prompt.md next to this file.
Needs `pip install anthropic` and ANTHROPIC_API_KEY (or `ant auth login`).

Latency notes:
  * effort=low keeps the answer short and the first token fast, which is what
    a voice reply needs; raise it for harder questions.
  * Server-side refusal fallbacks are enabled so a declined request is
    answered by another model instead of leaving the face silent.
"""

from __future__ import annotations

import argparse
import os
import sys

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))


def load_system_prompt() -> str:
    with open(os.path.join(HERE, "system_prompt.md"), encoding="utf-8") as f:
        text = f.read()
    # Everything below the "## System Prompt" heading *line* (the notes above it
    # mention the heading in passing, so match the line, not the phrase).
    import re
    m = re.search(r"^## System Prompt\s*$", text, re.M)
    return text[m.end():].strip() if m else text


def stream_reply(prompt: str, model: str, effort: str, character: str | None) -> None:
    from claude_chat import make_client
    client = make_client()
    system = load_system_prompt()
    if character:
        system += f"\n\nCharacter: {character}"
    with client.beta.messages.stream(
        model=model,
        max_tokens=1024,
        system=system,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            sys.stdout.write(text)
            sys.stdout.flush()
        final = stream.get_final_message()
    sys.stdout.write("\n")
    sys.stdout.flush()
    if final.stop_reason == "refusal":
        print("[claude] request was refused", file=sys.stderr)


def main() -> int:
    sys.path.insert(0, os.path.dirname(HERE))
    sys.path.insert(0, HERE)
    from env_config import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Stream a Claude reply to stdout for talker.py --stdin")
    p.add_argument("prompt", nargs="?", help="User message (reads stdin if omitted)")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--character", default=None,
                   help='Extra persona line, e.g. "a grumpy jack-o-lantern"')
    args = p.parse_args()
    prompt = args.prompt if args.prompt else sys.stdin.read().strip()
    if not prompt:
        p.error("no prompt given")
    try:
        stream_reply(prompt, args.model, args.effort, args.character)
    except anthropic.AuthenticationError:
        print("[claude] invalid or missing API key (set ANTHROPIC_API_KEY)", file=sys.stderr)
        return 1
    except anthropic.RateLimitError as e:
        print(f"[claude] rate limited: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"[claude] API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("[claude] network error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
