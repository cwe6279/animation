import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talker.stt_backends import EnergyEndpointer, STTBackend, Transcript
from talker.voice_loop import VoiceLoop
import numpy as np


class ScriptedSTT(STTBackend):
    """Returns queued transcripts one per feed(); tracks resets."""
    def __init__(self):
        self.queue = []
        self.resets = 0
        self.fed = 0
        self.speech_active = False

    def feed(self, pcm):
        self.fed += 1
        return self.queue.pop(0) if self.queue else None

    def reset(self):
        self.resets += 1
        self.speech_active = False


class FakeSpeaker:
    def __init__(self):
        self.spoken = []
        self.interrupts = 0
        self.busy = False

    def speak_stream(self, chunks):
        text = "".join(chunks)
        self.spoken.append(text)

    def interrupt(self):
        self.interrupts += 1
        self.busy = False

    @property
    def is_busy(self):
        return self.busy


def wait(pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_turn_runs_llm_and_speaks():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    events = []
    loop = VoiceLoop(stt, lambda t: iter(["[happy]You said ", t, "."]), spk,
                     on_event=lambda k, s: events.append((k, s)))
    stt.queue = [Transcript("hel", False), Transcript("hello", False), Transcript("hello there", True)]
    for _ in range(3):
        loop._process(b"\x00" * 320)
    assert wait(lambda: spk.spoken)
    assert spk.spoken == ["[happy]You said hello there."]
    kinds = [k for k, _ in events]
    assert kinds[:3] == ["hearing", "hearing", "you"]
    assert ("bot", "[happy]You said hello there.") in events
    assert loop.turns == 1


def test_mic_ignored_while_speaking_and_during_grace():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, on_event=lambda k, s: None)
    spk.busy = True
    stt.queue = [Transcript("echo of the speaker", True)]
    loop._process(b"\x00" * 320)
    assert stt.fed == 0 and stt.resets == 1 and spk.spoken == []
    spk.busy = False
    loop._process(b"\x00" * 320)          # still inside the grace window
    assert stt.fed == 0
    loop.GRACE_AFTER_SPEECH = 0.0
    loop._process(b"\x00" * 320)
    assert stt.fed == 1
    assert wait(lambda: spk.spoken == ["x"])


def test_barge_in_needs_sustained_speech():
    class Clk:
        t = 0.0
        def __call__(self): return self.t
    clk = Clk()
    stt, spk = ScriptedSTT(), FakeSpeaker()
    events = []
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, barge_in=True, barge_in_ms=500, clock=clk,
                     on_event=lambda k, s: events.append(k))
    spk.busy = True
    stt.speech_active = True
    loop._process(b"\x00" * 320)            # a blip: not yet
    assert spk.interrupts == 0
    clk.t += 0.2; loop._process(b"\x00" * 320)
    assert spk.interrupts == 0
    stt.speech_active = False               # the blip ended: counter resets
    loop._process(b"\x00" * 320)
    stt.speech_active = True
    clk.t += 0.1; loop._process(b"\x00" * 320)
    clk.t += 0.6; loop._process(b"\x00" * 320)   # sustained talking over her
    assert spk.interrupts == 1 and "barge-in" in events


def test_llm_failure_speaks_apology():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    events = []

    def broken(t):
        raise RuntimeError("no api")
        yield
    loop = VoiceLoop(stt, broken, spk, on_event=lambda k, s: events.append((k, s)))
    loop.on_user_text("hi")
    assert wait(lambda: spk.spoken)
    assert spk.spoken[0].startswith("[sad]Sorry")
    assert any(k == "error" for k, _ in events)
    assert not loop._thinking


def test_second_utterance_while_thinking_is_dropped():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    gate = threading.Event()

    def slow(t):
        gate.wait(2)
        yield "done"
    loop = VoiceLoop(stt, slow, spk, on_event=lambda k, s: None)
    loop.on_user_text("first")
    loop.on_user_text("second")
    gate.set()
    assert wait(lambda: spk.spoken)
    assert loop.turns == 1


def test_process_hands_off_to_worker_thread():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, on_event=lambda k, s: None)
    stt.queue = [Transcript("hello there", True)]
    loop.process(b"\x00" * 320)          # returns immediately; worker does the rest
    assert wait(lambda: spk.spoken == ["x"])


def test_energy_endpointer_detects_utterance():
    sr = 16000
    ep = EnergyEndpointer(sr, silence_ms=200)
    quiet = (np.random.randn(1600) * 50).astype(np.int16).tobytes()      # 100 ms
    loud = (np.sin(np.arange(1600) * 0.3) * 8000).astype(np.int16).tobytes()
    for _ in range(8):                       # calibration + quiet
        assert ep.feed(quiet) is None
    assert ep.feed(loud) is None             # onset needs two consecutive loud frames
    assert ep.feed(loud) is None and ep.active
    out = None
    for _ in range(4):
        out = out or ep.feed(quiet)
    assert out is not None and len(out) >= 2 * 1600 * 2
    assert not ep.active


def test_talker_app_satisfies_speaker_interface():
    """VoiceLoop drives TalkerApp; make sure the app exposes what the loop calls."""
    import inspect
    import talker.app as talker
    for name in ("speak_stream", "interrupt", "is_busy"):
        assert hasattr(talker.TalkerApp, name), name
    assert isinstance(inspect.getattr_static(talker.TalkerApp, "is_busy"), property)


def test_energy_endpointer_handles_noisy_room():
    """Room noise ~2000 RMS (a hot USB mic) must not read as speech; speech at ~15000 must."""
    sr = 16000
    rng = np.random.default_rng(0)
    ep = EnergyEndpointer(sr, silence_ms=400)
    frame = 1024
    noise = lambda: (rng.standard_normal(frame) * 2000).astype(np.int16).tobytes()
    speech = lambda: (rng.standard_normal(frame) * 15000).astype(np.int16).tobytes()
    for _ in range(40):                       # 2.5 s of noise: never starts
        assert ep.feed(noise()) is None
        assert not ep.active
    assert 1500 < ep.floor < 2600
    for _ in range(20):                       # ~1.3 s of speech
        assert ep.feed(speech()) is None
    assert ep.active
    out = None
    for _ in range(12):                       # back to noise: ends within ~0.8 s
        out = out or ep.feed(noise())
    assert out is not None and not ep.active
    assert len(out) // 2 >= 20 * frame        # the speech is in there


def test_echo_guard_tells_echo_from_a_person():
    import math
    from talker.audio_engine import EchoGuard
    # speaker envelope: a 3 Hz syllable rhythm
    out = [(t / 100.0, 1000 + 800 * math.sin(2 * math.pi * 3 * t / 100.0)) for t in range(0, 150)]
    echo = EchoGuard()
    for t, v in out:
        echo.add_mic(t + 0.06, v * 0.3 + 50)                 # same rhythm, 60 ms late, quieter
    assert echo.correlation(out, 1.5) > 0.8
    person = EchoGuard()
    for t in range(0, 150):
        person.add_mic(t / 100.0, 1500 + 700 * math.sin(2 * math.pi * 1.3 * t / 100.0 + 1.0))   # a different rhythm
    assert person.correlation(out, 1.5) < 0.5


def test_barge_in_ignores_a_knock_but_not_sustained_loudness():
    """A knock trips the endpointer, whose active flag lingers through its silence gate;
    the barge-in must still see the mic stay loud for most of the window."""
    import numpy as np

    class Clk:
        t = 0.0
        def __call__(self): return self.t
    clk = Clk()
    stt, spk = ScriptedSTT(), FakeSpeaker()
    stt._ep = type("Ep", (), {"floor": 100.0, "min_rms": 100.0, "start_ratio": 3.0, "gate_boost": 4.0})()
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, barge_in=True, barge_in_ms=500, clock=clk)
    spk.busy = True
    quiet = np.zeros(320, np.int16).tobytes()
    loud = (np.ones(320, np.int16) * 5000).tobytes()
    stt.speech_active = True                     # the endpointer latched on a knock...
    loop._process(loud)
    for _ in range(30):                          # ...and stays 'active' through quiet frames
        clk.t += 0.02; loop._process(quiet)
    assert spk.interrupts == 0                   # 600 ms of mostly silence: not a barge-in
    for _ in range(30):                          # a person talking over it: loud frames throughout
        clk.t += 0.02; loop._process(loud)
    assert spk.interrupts == 1


def test_waiting_mode_hears_nothing_and_stops_talking():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    events = []
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, on_event=lambda k, s: events.append((k, s)))
    spk.busy = True
    loop.set_waiting(True, "spacebar")
    assert spk.interrupts == 1 and ("mode", "waiting (spacebar)") in events
    stt.queue = [Transcript("hello there", True)]
    loop._process(b"\x00" * 320)                 # audio is dropped, the transcript never surfaces
    assert stt.queue and not any(k == "hearing" for k, _ in events)
    loop.on_user_text("hi")                      # typed text too
    assert not spk.spoken and any(k == "ignored" for k, _ in events)
    loop.set_waiting(False)
    assert ("mode", "listening again") in events


def test_echo_guard_finds_echo_that_arrives_half_a_second_late():
    """Output buffer + room + input buffer can add several hundred ms; the guard must
    still line the two envelopes up."""
    import math
    from talker.audio_engine import EchoGuard
    out = [(t / 100.0, 1000 + 800 * math.sin(2 * math.pi * 3 * t / 100.0)) for t in range(0, 200)]
    echo = EchoGuard()
    for t, v in out:
        echo.add_mic(t + 0.45, v * 0.3 + 50)
    assert echo.correlation(out, 2.4) > 0.8


def test_mic_frames_are_stamped_when_heard_not_when_processed():
    """A slow recognizer must not skew the mic timeline the echo guard compares."""
    import numpy as np
    stt, spk = ScriptedSTT(), FakeSpeaker()
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, barge_in=True)
    spk.busy = True
    loud = (np.ones(320, np.int16) * 5000).tobytes()
    loop._process(loud, t_in=123.456)
    assert loop._echo._mic[-1][0] == 123.456
    loop.process(loud)                                   # the callback path stamps and queues
    t_in, pcm = loop._audio_q.get_nowait()
    assert pcm == loud and abs(t_in - time.monotonic()) < 0.5


def test_an_echo_verdict_holds_while_the_correlation_wobbles():
    """Her own voice scored 0.62 then 0.43 four hundred ms later and barged in on her."""
    import numpy as np

    class Clk:
        t = 100.0
        def __call__(self): return self.t
    clk = Clk()
    stt, spk = ScriptedSTT(), FakeSpeaker()
    stt._ep = type("Ep", (), {"floor": 100.0, "min_rms": 100.0, "start_ratio": 3.0, "gate_boost": 4.0})()
    spk.output_envelope = lambda: [(0, 1)]
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, barge_in=True, barge_in_ms=400, clock=clk)
    scores = iter([0.62, 0.43, 0.43, 0.43, 0.20, 0.20, 0.20, 0.20])
    loop._echo.correlation = lambda env, now: next(scores)
    spk.busy = True
    stt.speech_active = True
    loud = (np.ones(320, np.int16) * 5000).tobytes()
    t = 100.0
    def talk(seconds):
        nonlocal t
        for _ in range(int(seconds / 0.02)):
            t += 0.02; clk.t = t
            loop._process(loud, t_in=t)
    talk(0.45)                         # first verdict: 0.62, echo
    assert spk.interrupts == 0
    talk(0.9)                          # 0.43 twice within the hold: still echo
    assert spk.interrupts == 0
    talk(1.6)                          # the hold has lapsed and the mic no longer follows her: a person
    assert spk.interrupts == 1


def test_echo_transcript_is_dropped_even_after_a_barge_in():
    stt, spk = ScriptedSTT(), FakeSpeaker()
    heard = []
    loop = VoiceLoop(stt, lambda t: (heard.append(t), iter(["x"]))[1], spk)
    loop._last_said = "[annoyed] Because you always forget it. Or perhaps you think it is clever to mock me."
    loop._spoke_at = time.monotonic()
    loop._last_busy = 0.0              # what a barge-in does
    loop.on_user_text("Because you all.")
    time.sleep(0.05)
    assert heard == []


def test_barge_in_ignores_a_clap():
    """A clap is loud for a few frames and rings down; at 400 ms it still passed the duty
    share. Speech is still loud when the window closes; a clap is not."""
    import numpy as np

    class Clk:
        t = 0.0
        def __call__(self): return self.t
    clk = Clk()
    stt, spk = ScriptedSTT(), FakeSpeaker()
    stt._ep = type("Ep", (), {"floor": 100.0, "min_rms": 100.0, "start_ratio": 3.0, "gate_boost": 4.0})()
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, barge_in=True, barge_in_ms=400, clock=clk)
    spk.busy = True
    stt.speech_active = True
    quiet = np.zeros(320, np.int16).tobytes()
    loud = (np.ones(320, np.int16) * 5000).tobytes()
    for _ in range(4):                           # 160 ms of clap: over the 0.4 duty share on its own
        clk.t += 0.04; loop._process(loud)
    for _ in range(8):                           # ...then the room rings down
        clk.t += 0.04; loop._process(quiet)
    assert spk.interrupts == 0
    for _ in range(12):                          # a person: loud right through
        clk.t += 0.04; loop._process(loud)
    assert spk.interrupts == 1


def test_barge_in_waits_until_the_echo_guard_has_something_to_judge():
    """At her first words the guard has no history and scored 0.0, which read as a person."""
    import numpy as np

    class Clk:
        t = 50.0
        def __call__(self): return self.t
    clk = Clk()
    stt, spk = ScriptedSTT(), FakeSpeaker()
    stt._ep = type("Ep", (), {"floor": 100.0, "min_rms": 100.0, "start_ratio": 3.0, "gate_boost": 4.0})()
    spk.output_envelope = lambda: [(0, 1)]
    loop = VoiceLoop(stt, lambda t: iter(["x"]), spk, barge_in=True, barge_in_ms=400, clock=clk)
    loop._echo.correlation = lambda env, now: 0.0        # "a person", if asked
    spk.busy = True
    stt.speech_active = True
    loud = (np.ones(320, np.int16) * 5000).tobytes()
    t = 50.0
    for _ in range(30):                                  # 600 ms: the duty and window are satisfied...
        t += 0.02; clk.t = t; loop._process(loud, t_in=t)
    assert spk.interrupts == 0                           # ...but the guard cannot judge yet
    for _ in range(15):                                  # 900 ms in: it can, and it says a person
        t += 0.02; clk.t = t; loop._process(loud, t_in=t)
    assert spk.interrupts == 1
