"""Action blocks: parsed out of the speech, fired on the word timeline."""
import os
import time

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from talker.actions import Action, ActionDispatcher, ToolBox, action_rules, parse_actions
from talker.audio_engine import NullAudioEngine
from talker.phoneme_scheduler import ScheduleReader, parse_tags
from talker.speech_pipeline import SpeechPipeline
from tests.test_pipeline import FakeBackend, wait_until


def test_parse_actions_strips_blocks_and_counts_spoken_words():
    text = "[happy] Welcome in! {{move nod}} Mind the step. {{sfx creak}}"
    clean, acts = parse_actions(text)
    assert "{{" not in clean and "[happy]" in clean          # emotion tags stay for the voice
    assert parse_tags(clean)[0] == "Welcome in! Mind the step."
    assert [(i, a.kind, a.name) for i, a in acts] == [(2, "move", "nod"), (5, "sfx", "creak")]


def test_parse_actions_args_and_json():
    _, acts = parse_actions('Sure. {{tool weather Denver, CO}} {{move: turn {"deg": 30}}}')
    assert acts[0][1].args == "Denver, CO" and acts[0][1].params == {"text": "Denver, CO"}
    assert acts[1][1].name == "turn" and acts[1][1].params == {"deg": 30}
    assert parse_actions("no blocks here") == ("no blocks here", [])


def test_dispatcher_routes_and_feeds_tool_results_back():
    seen = []
    tools = ToolBox()
    tools.add("clock", "what time it is", lambda args: "3 pm")
    d = ActionDispatcher(on_result=lambda a, r: seen.append((a.name, r)))
    d.register("tool", tools.handler())
    d.dispatch(Action("tool", "clock", raw="{{tool clock}}"))
    d.dispatch(Action("move", "nod", raw="{{move nod}}"))      # no handler: ignored, no error
    assert seen == [("clock", "3 pm")]
    assert "clock" in tools.describe()


def test_action_rules_only_mention_what_exists():
    assert action_rules() == ""
    r = action_rules(moves=["nod"], sounds=["creak"])
    assert "{{move" in r and "creak" in r and "{{tool" not in r


def test_pipeline_fires_actions_when_their_words_are_spoken():
    eng = NullAudioEngine(sample_rate=1000)
    p = SpeechPipeline(eng, ScheduleReader(), FakeBackend(), lead_seconds=0.0)
    fired = []
    p.on_action = lambda a: fired.append((a.name, time.monotonic()))
    p.start()
    try:
        p.speak("Hello there {{move nod}} friend. {{sfx creak}}")   # 0.2 s per word
        assert wait_until(lambda: len(fired) == 2, timeout=4.0), fired
    finally:
        p.stop()
    names = [n for n, _ in fired]
    assert names == ["nod", "creak"]
    t_first = p.first_audio_at
    assert 0.3 <= fired[0][1] - t_first <= 0.9      # "nod" sits before word 3 (0.4 s in)
    assert fired[1][1] >= fired[0][1]               # trailing block fires at the end
    assert "{{" not in " ".join(p.backend.seen)     # the voice never saw a block


def test_sfx_waits_for_the_character_to_stop_talking():
    """A sound effect written mid-sentence is heard after the sentence, not over it."""
    from talker.actions import SoundBank
    bank = SoundBank(None)
    played, speaking = [], {"busy": True}
    bank.play = lambda name: (played.append(name), True)[1]
    handler = bank.handler(hold_while=lambda: speaking["busy"], ends_at=lambda: 0.05)

    handler(Action("sfx", "meow", raw="{{sfx meow}}"))
    assert played == []                                  # she is talking: held
    speaking["busy"] = False
    assert wait_until(lambda: played == ["meow"], timeout=2), played

    speaking["busy"] = True                              # still talking when the timer fires: dropped
    handler(Action("sfx", "purr", raw="{{sfx purr}}"))
    time.sleep(0.4)
    assert played == ["meow"]

    quiet = bank.handler()                               # no hold: the old overlapping behaviour
    quiet(Action("sfx", "hiss", raw="{{sfx hiss}}"))
    assert played == ["meow", "hiss"]
