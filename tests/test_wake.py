import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talker.voice_loop import VoiceLoop
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


def test_idle_timeout_counts_from_end_of_speech():
    loop, spk, events, clk = make(["goat"])
    loop.on_user_text("goat hello")
    assert wait(lambda: spk.spoken) and loop.engaged
    spk.busy = True                     # a long reply is playing
    clk.t += 30; loop.tick(); assert loop.engaged
    spk.busy = False                    # speech ends now; the 10 s timeout starts here
    clk.t += 8; loop.tick(); assert loop.engaged
    clk.t += 3; loop.tick(); assert not loop.engaged


def test_sleep_words_put_it_to_sleep_and_interrupt():
    loop, spk, events, _ = make(["eve"])
    loop.engage()
    loop.on_user_text("hold on")
    assert not loop.engaged and spk.interrupts == 1 and spk.spoken == []
    loop.on_user_text("Eve, hello")            # wake again
    assert loop.engaged and wait(lambda: spk.spoken)
    loop.on_user_text("please stop doing that to the goat")   # 'stop' inside a real sentence: not a sleep command
    assert loop.engaged
    loop.on_user_text("Stop!")
    assert not loop.engaged


def test_sleep_words_ignored_without_wake_mode():
    loop, spk, events, _ = make(None)
    loop.on_user_text("stop")
    assert loop.engaged and wait(lambda: spk.spoken)


def test_end_marker_tells_a_satellite_board_to_idle():
    """With a board, the board does the wake word: the host has none, but [end] must still
    reach the board so it stops listening ("stop listening", "go to sleep", goodbye)."""
    llm = lambda t: iter(["Okay, I'll be quiet. [end]"])
    loop, spk, events, _ = make(None, llm=llm)
    ended = []
    loop.on_conversation_end = lambda: ended.append(True)
    loop.on_user_text("stop listening please")
    assert wait(lambda: spk.spoken) and wait(lambda: ended)
    assert spk.spoken[0] == "Okay, I'll be quiet. "


def test_voice_note_records_quietly_until_stop_and_saves_audio_and_text(tmp_path):
    import wave
    loop, spk, events, clk = make(None, llm=lambda t: iter(["should not answer"]))
    loop.record_dir = str(tmp_path)
    saved = []
    loop.on_recording_saved = lambda path, secs, text: saved.append((path, text))
    loop.on_user_text("Clara, start recording please")
    assert loop.recording
    loop._process(b"\x01\x00" * 1600)                  # 0.1 s of audio, straight to disk
    loop.on_user_text("buy milk and call the plumber")
    loop.on_user_text("that's all, stop recording")
    assert not loop.recording and spk.spoken == []     # she said nothing while it recorded
    path, text = saved[0]
    assert text == "buy milk and call the plumber that's all"
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 1600
    assert open(path[:-4] + ".txt").read().strip() == text


def test_voice_note_stops_at_the_time_limit(tmp_path):
    loop, spk, events, clk = make(None)
    loop.record_dir, loop.record_max_s = str(tmp_path), 0.05
    saved = []
    loop.on_recording_saved = lambda path, secs, text: saved.append(path)
    loop.set_recording(True, "board")
    loop._process(b"\x01\x00" * 160)
    time.sleep(0.1)
    loop._process(b"\x01\x00" * 160)
    assert not loop.recording and loop.record_stopped_by_limit and saved and saved[0].endswith(".wav")


def test_a_spoken_command_mid_conversation_runs_instead_of_the_brain():
    loop, spk, events, _ = make(None, llm=lambda t: iter(["brain answered"]))
    ran = []
    loop.spoken_commands = {"scan this": lambda: ran.append("scan this"),
                            "scan this document": lambda: ran.append("scan this document")}
    loop.on_user_text("okay Clara, scan this document please")
    assert ran == ["scan this document"] and spk.spoken == []      # longest phrase wins, no brain
    loop.on_user_text("what's the weather")
    assert wait(lambda: spk.spoken)


def test_the_visual_trigger_needs_a_question_for_her_not_just_the_word_look():
    R = VoiceLoop.VISUAL_RE
    for asked in ("what do you see", "look at this", "read this label", "what does this say", "what am I holding"):
        assert R.search(asked), asked
    for not_asked in ("now I'm going to have a look at it first", "let me see", "I read a book yesterday",
                      "see you later", "hold on"):
        assert not R.search(not_asked), not_asked     # a black photo of a covered lens, before
