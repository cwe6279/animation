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
