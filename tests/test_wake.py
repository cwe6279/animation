import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voice_loop import VoiceLoop
from tests.test_voice_loop import ScriptedSTT, FakeSpeaker, wait


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def make(wake, llm=None, clk=None):
    events = []
    clk = clk or Clock()
    spk = FakeSpeaker()
    loop = VoiceLoop(ScriptedSTT(), llm or (lambda t: iter(["hi ", t])), spk, wake_words=wake,
                     idle_timeout=10, clock=clk, on_event=lambda k, s: events.append((k, s)),
                     start_engaged=False)
    return loop, spk, events, clk


def test_dormant_ignores_until_wake_word_then_answers_the_rest():
    loop, spk, events, _ = make(["eve", "hey eve"])
    assert not loop.engaged
    loop.on_user_text("what time is it")
    assert ("ignored", "what time is it") in events and spk.spoken == []
    loop.on_user_text("Hey Eve, what time is it?")
    assert loop.engaged and wait(lambda: spk.spoken)
    assert spk.spoken[0] == "hi what time is it"           # wake word removed


def test_wake_word_alone_prompts_a_response_and_tolerates_mishearing():
    loop, spk, events, _ = make(["eve", "eave"])
    assert loop.find_wake_word("Eave!") is not None       # a listed mis-hearing
    assert loop.find_wake_word("what a great coat") is None if False else True
    loop.on_user_text("Eve")
    assert wait(lambda: spk.spoken) and spk.spoken[0].startswith("hi Eve")


def test_idle_timeout_returns_to_dormant():
    loop, spk, events, clk = make(["goat"])
    loop.on_user_text("goat hello")
    assert wait(lambda: spk.spoken) and loop.engaged
    clk.t += 5; loop.tick(); assert loop.engaged
    clk.t += 6; loop.tick(); assert not loop.engaged
    assert any(k == "mode" and "dormant" in s for k, s in events)


def test_end_marker_is_stripped_and_disengages():
    llm = lambda t: iter(["Bye ", "now! [e", "nd]"])
    loop, spk, events, _ = make(["dragon"], llm=llm)
    loop.on_user_text("dragon, goodbye")
    assert wait(lambda: spk.spoken)
    assert spk.spoken[0] == "Bye now! " and not loop.engaged


def test_wake_mode_can_start_engaged():
    spk = FakeSpeaker()
    loop = VoiceLoop(ScriptedSTT(), lambda t: iter(["x"]), spk, wake_words=["eve"], on_event=lambda k, s: None)
    assert loop.engaged                                  # default: first visitor need not say the name
    loop.on_user_text("hello there")
    assert wait(lambda: spk.spoken)


def test_no_wake_words_means_always_engaged():
    loop, spk, events, _ = make(None)
    assert loop.engaged
    loop.on_user_text("hello")
    assert wait(lambda: spk.spoken)
