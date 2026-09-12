"""The microphone filters: do they pass speech, reject rumble, and survive chunking?"""
import numpy as np

from talker.audio_filters import Biquad, MicFilter

RATE = 16000


def tone(hz, seconds=0.5, rate=RATE, amp=8000.0):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def level(pcm_bytes):
    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    half = a[a.size // 2:]                      # skip the filter's settling transient
    return float(np.sqrt(np.mean(half * half)))


def db(out, ref):
    return 20.0 * np.log10(max(out, 1e-9) / max(ref, 1e-9))


def test_high_pass_kills_rumble_and_leaves_speech_alone():
    """One section is 12 dB per octave, so 20 Hz is buried, 40 Hz is well down,
    the corner sits at -3 dB by definition, and speech is untouched."""
    for hz, floor_db, ceil_db in ((20, -60, -22),      # mains hum and building rumble
                                  (40, -22, -10),      # air conditioning, a desk thump
                                  (90, -4.0, -2.0),    # the corner itself
                                  (300, -1.0, 1.0),    # the bottom of a voice
                                  (1000, -1.0, 1.0)):  # the middle of the speech band
        src = tone(hz)
        out = Biquad.high_pass(90.0, RATE).process_bytes(src.tobytes())
        got = db(level(out), level(src.tobytes()))
        assert floor_db <= got <= ceil_db, f"{hz} Hz moved {got:.1f} dB"


def test_low_pass_removes_what_would_alias():
    src = tone(7000)                            # above half of a 8 kHz target rate
    out = Biquad.low_pass(3500.0, RATE).process_bytes(src.tobytes())
    assert db(level(out), level(src.tobytes())) < -20


def test_filtering_in_chunks_matches_filtering_it_whole():
    """State has to carry across audio callbacks or every chunk boundary clicks."""
    src = tone(300).tobytes()
    whole = Biquad.high_pass(90.0, RATE).process_bytes(src)
    f = Biquad.high_pass(90.0, RATE)
    step = 1024 * 2                             # 1024 int16 samples per "callback"
    pieces = b"".join(f.process_bytes(src[i:i + step]) for i in range(0, len(src), step))
    assert pieces == whole


def test_zero_disables_a_stage_and_empty_input_is_safe():
    f = MicFilter(RATE, high_pass=0, low_pass=0)
    assert not f.active and f.describe() == "off"
    src = tone(40).tobytes()
    assert f.process_bytes(src) == src          # untouched
    assert f.process_bytes(b"") == b""


def test_cutoffs_can_change_while_running():
    f = MicFilter(RATE, high_pass=90, low_pass=0)
    assert "high-pass 90 Hz" in f.describe()
    quiet = level(f.process_bytes(tone(40).tobytes()))
    f.configure(high_pass=300)                  # what the control page does
    assert "high-pass 300 Hz" in f.describe()
    quieter = level(f.process_bytes(tone(40).tobytes()))
    assert quieter < quiet                      # a higher corner rejects 40 Hz harder


def test_cutoff_above_nyquist_is_clamped_not_broken():
    f = Biquad.low_pass(999_999, RATE)
    assert f.cutoff < RATE / 2
    out = f.process_bytes(tone(1000).tobytes())
    assert level(out) > 1000                    # still passes audio rather than exploding
