import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talker.calibrate import recommend


def test_recommend_good_separation():
    r = recommend(ambient=300, speaker=800, person=5000)
    assert r["barge_in_ok"] and r["mic_gain"] == "ok"
    thr = r["onset_threshold"]
    assert 800 < r["barge_in_boost"] * thr < 5000            # threshold lands between bleed and person


def test_recommend_marginal_and_bad():
    assert not recommend(300, 3000, 4500)["barge_in_ok"]
    assert "not viable" in recommend(300, 4000, 4200)["verdict"]
    assert recommend(300, 500, 300)["mic_gain"] == "raise"
    assert recommend(300, 500, 30000)["mic_gain"] == "lower"


def test_boost_is_clamped():
    assert 1.0 <= recommend(50, 100, 200)["barge_in_boost"] <= 8.0
    assert recommend(5000, 100000, 200000)["barge_in_boost"] <= 8.0


def test_open_timeout_and_device_fallback_list():
    """A driver that never returns must not hang start-up, and a name that matches
    several devices must offer all of them so the caller can try the next."""
    import pytest
    from talker.audio_engine import AudioEngine

    class Wedged(AudioEngine):
        OPEN_TIMEOUT_S = 0.2

        def __init__(self):                      # no PortAudio in the test environment
            self._pa = None

        def list_input_devices(self):
            return [(7, "Samson Go Mic: USB Audio (hw:4,0)", 44100, False),
                    (14, "GoMic compact condenser mic Analog Stereo", 48000, False)]

    a = Wedged()
    assert a.resolve_devices("gomic", "input") == [14]
    assert a.resolve_devices("go mic", "input") == [7]        # raw device, the only match
    assert a.resolve_devices(None, "input") == [None]
    assert a.resolve_devices(7, "input") == [7]
    with pytest.raises(RuntimeError):
        a.resolve_devices("nosuchmic", "input")

    import time
    a._pa = type("PA", (), {"open": staticmethod(lambda **kw: time.sleep(30))})()
    t = time.monotonic()
    with pytest.raises(TimeoutError):
        a._open_stream(rate=48000)
    assert time.monotonic() - t < 3            # gave up instead of blocking
