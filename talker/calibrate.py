"""
calibrate.py — a guided check of a new mic/speaker setup.

    python voice_loop.py --calibrate --mic-device gomic --output-device jabra

Three measurements, a few seconds each:
  1. the room with nobody talking          -> ambient level
  2. the character speaking through the speaker -> speaker bleed into the mic
  3. you speaking from the visitor spot    -> person level
From those it decides whether barge-in can work, picks --barge-in-boost, flags
mic gain problems, and writes calibration.json (gitignored, per machine). The
voice loop reads that file for its defaults when the flags are not given.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_FILE = os.path.join(ROOT, "calibration.json")

# The onset threshold the endpointer uses: max(min_rms, floor * start_ratio); see stt_backends.
MIN_RMS, START_RATIO = 350.0, 3.0


def recommend(ambient: float, speaker: float, person: float) -> Dict:
    """
    Turn the three levels into settings.
      barge_in_ok  : the person is clearly louder than the speaker bleed (>= 2x)
      boost        : puts the barge-in threshold at the geometric mean of bleed and person
      gain         : 'raise' / 'lower' / 'ok' for the mic input gain
    """
    threshold = max(MIN_RMS, ambient * START_RATIO)
    ratio = person / max(1.0, speaker)
    target = float(np.sqrt(max(1.0, speaker) * max(1.0, person)))
    boost = target / threshold
    boost = float(min(8.0, max(1.0, round(boost, 1))))
    if person < 500:
        gain = "raise"
    elif person > 24000:
        gain = "lower"
    else:
        gain = "ok"
    verdict = ("barge-in ok" if ratio >= 2.0 else
               "barge-in marginal: move the speaker away from or behind the mic, or lower the volume" if ratio >= 1.3 else
               "barge-in not viable with this placement: use half-duplex (the default)")
    return {"ambient": round(ambient), "speaker": round(speaker), "person": round(person),
            "person_over_speaker": round(ratio, 2), "onset_threshold": round(threshold),
            "barge_in_ok": ratio >= 2.0, "barge_in_boost": boost, "mic_gain": gain, "verdict": verdict}


def _measure(audio, seconds: float, label: str, on_level: Optional[Callable[[float], None]] = None) -> float:
    """Average mic RMS over `seconds`, sampling 10 times a second (ignores the quietest 20%)."""
    vals = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        time.sleep(0.1)
        rms, _ = audio.get_state()
        vals.append(rms)
        if on_level:
            on_level(rms)
    vals.sort()
    keep = vals[len(vals) // 5:] or vals
    return float(np.mean(keep)) if keep else 0.0


def run_calibration(audio, speak: Callable[[str], None], is_speaking: Callable[[], bool],
                    mic_device, output_device, text: str = "Testing one two three. Can you hear me from the door?",
                    ask: Callable[[str], None] = None) -> Dict:
    ask = ask or (lambda msg: input(msg))
    print("\n== Calibration ==")
    print("1/3  Room level. Keep quiet for 4 seconds...")
    ambient = _measure(audio, 4.0, "ambient")
    print(f"     ambient {ambient:.0f}")

    print("2/3  Speaker bleed. The character will talk through the speaker; stay quiet.")
    speak(" ".join([text] * 2))
    t0 = time.monotonic()
    while not is_speaking() and time.monotonic() - t0 < 15:
        time.sleep(0.05)
    vals = []
    while is_speaking():
        time.sleep(0.1)
        vals.append(audio.get_state()[0])
    vals.sort()
    speaker = float(np.mean(vals[len(vals) // 2:])) if vals else 0.0      # the louder half of the reply
    print(f"     speaker bleed {speaker:.0f}")

    ask("3/3  Stand where visitors will stand and press Enter, then talk normally for 5 seconds... ")
    person = _measure(audio, 5.0, "person")
    print(f"     person {person:.0f}")

    rec = recommend(ambient, speaker, person)
    rec.update({"mic_device": mic_device, "output_device": output_device,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print("\n== Result ==")
    print(f"  ambient {rec['ambient']}, speaker bleed {rec['speaker']}, person {rec['person']} "
          f"(person is {rec['person_over_speaker']}x the speaker)")
    print(f"  {rec['verdict']}")
    print(f"  recommended --barge-in-boost {rec['barge_in_boost']}")
    if rec["mic_gain"] != "ok":
        print(f"  mic gain: {'too low, raise it in the sound settings' if rec['mic_gain'] == 'raise' else 'too hot, lower it to avoid clipping'}")
    print(f"  saved to {CALIBRATION_FILE}; voice_loop.py uses these defaults when flags are not given")
    return rec


def load_calibration() -> Optional[Dict]:
    try:
        with open(CALIBRATION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
