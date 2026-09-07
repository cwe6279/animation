"""
tools/demo_emotions.py — Demo script that cycles through emotion tags.

Launches the talker with a series of sentences showcasing each emotion.
Each demo auto-exits after speech finishes. Press ESC to skip early.
Press Ctrl+C in the terminal to stop all demos.
"""

import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

import subprocess
import sys
import os

FACE = "cat"
PYTHON = sys.executable

# Each entry: (description, text with inline emotion tags)
DEMOS = [
    ("Neutral (default)",
     "Hello, I am speaking normally with no emotion tags."),

    ("Happy",
     "[happy]I'm so happy to see you! This is wonderful!"),

    ("Angry",
     "[angry]I am very angry right now! This is unacceptable!"),

    ("Annoyed",
     "[annoyed]Ugh, this is getting really annoying. Can you stop?"),

    ("Sad",
     "[sad]I feel so sad today. Everything seems gloomy."),

    ("Surprise",
     "[surprise]Oh wow! I did not expect that at all!"),

    ("Mixed emotions mid-sentence",
     "[happy]I was having a great day, [angry]but then someone cut me off in traffic! "
     "[sad]It really ruined my mood. [neutral]Anyway, I'm over it now."),

    ("Angry to happy",
     "[angry]Stop doing that! [happy]Actually, you know what, it's fine. I'm happy!"),
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    talker_py = os.path.join(ROOT, "speak.py")

    print("=" * 60)
    print("  EMOTION TAG DEMO")
    print("  Face: " + FACE)
    print("  Each demo auto-exits after speech. ESC to skip early.")
    print("=" * 60)

    for i, (desc, text) in enumerate(DEMOS, 1):
        print(f"\n--- Demo {i}/{len(DEMOS)}: {desc} ---")
        print(f"    Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        cmd = [
            PYTHON, talker_py,
            "--face", FACE,
            "--text", text,
            "--debug",
            "--auto-exit",
        ]
        try:
            subprocess.run(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            print("\n[test] Interrupted — exiting")
            sys.exit(0)

    print("\n" + "=" * 60)
    print("  All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
