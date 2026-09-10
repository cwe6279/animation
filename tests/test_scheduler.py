import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from talker.phoneme_scheduler import (
    Emotion, EmotionEvent, ScheduleReader, SentenceSplitter, Viseme, VisemeEvent,
    arpabet_to_visemes, grapheme_to_arpabet_fallback, parse_emotion_tags,
    split_sentences, word_to_viseme_events, estimate_word_times,
)


def test_parse_emotion_tags_basic():
    clean, tags = parse_emotion_tags("[angry]I'm so mad! [sad]But also hurt.")
    assert clean == "I'm so mad! But also hurt."
    assert tags == [(0, Emotion.ANGRY), (3, Emotion.SAD)]


def test_parse_emotion_tags_unknown_and_stacked():
    clean, tags = parse_emotion_tags("Hey [pauses] there [happy][surprise]friend")
    assert clean == "Hey there friend"
    assert tags == [(2, Emotion.SURPRISE)]           # [pauses] has no face meaning; last stacked tag wins


def test_parse_tags_voice_text_and_vocabulary():
    from talker.phoneme_scheduler import parse_tags, tag_to_emotion
    clean, voiced, tags = parse_tags("[excited]Oh wow! [light chuckle] It was cute. [British accent] Right?")
    assert clean == "Oh wow! It was cute. Right?"
    assert voiced == "[excited] Oh wow! [light chuckle] It was cute. [British accent] Right?"
    assert tags == [(0, Emotion.HAPPY), (2, Emotion.HAPPY)]   # accent: voice only
    assert tag_to_emotion("nervously") is Emotion.SURPRISE
    assert tag_to_emotion("sigh of relief") is Emotion.HAPPY
    assert tag_to_emotion("resigned tone") is Emotion.SAD
    assert tag_to_emotion("frustrated") is Emotion.ANGRY
    assert tag_to_emotion("pirate voice") is None


def test_parse_emotion_tags_none():
    assert parse_emotion_tags("plain text") == ("plain text", [])
    assert parse_emotion_tags("") == ("", [])


def test_split_sentences_batch():
    out = split_sentences("[happy]I was great. [angry]Then not! Really? Yes.")
    assert out == ["[happy]I was great.", "[angry]Then not! Really?", "Yes."]


def test_split_sentences_merges_short_fragments():
    assert split_sentences("Oh. Well then, that is a surprise. Wow.") == \
        ["Oh. Well then, that is a surprise.", "Wow."]


def test_splitter_streaming_tokens():
    s = SentenceSplitter(min_chars=1)
    got = []
    for tok in ["Hel", "lo the", "re. Sec", "ond one!", " Third"]:
        got += s.feed(tok)
    got += s.flush()
    assert got == ["Hello there.", "Second one!", "Third"]


def test_fallback_g2p():
    assert grapheme_to_arpabet_fallback("chat") == ["CH", "AE", "T"]
    assert grapheme_to_arpabet_fallback("queen") == ["K", "W", "IY", "N"]


def test_arpabet_to_visemes_strips_stress_and_defaults():
    assert arpabet_to_visemes(["HH", "AH0", "L", "OW1", "ZZZ"]) == \
        [Viseme.AH, Viseme.AH, Viseme.DD, Viseme.OO, Viseme.AH]


def test_word_to_viseme_events_fills_window_and_dedupes():
    evs = word_to_viseme_events("mom", 1.0, 1.5)
    assert evs[0].time == pytest.approx(1.0)
    assert evs[-1].end_time == pytest.approx(1.5)
    shapes = [e.viseme for e in evs]
    assert len(shapes) == 3 and shapes[0] is Viseme.PP and shapes[2] is Viseme.PP
    assert shapes[1] in (Viseme.AA, Viseme.AH, Viseme.OO)
    # vowels get more time than consonants
    assert evs[1].duration > evs[0].duration


def test_word_to_viseme_events_min_duration():
    evs = word_to_viseme_events("a", 2.0, 2.0)
    assert evs[-1].end_time >= 2.05


def test_estimate_word_times_monotonic():
    wt = estimate_word_times("hello big world")
    assert [w for w, _, _ in wt] == ["hello", "big", "world"]
    assert all(wt[i][2] <= wt[i + 1][1] for i in range(len(wt) - 1))


def test_schedule_reader_gaps_and_append():
    r = ScheduleReader([VisemeEvent(1.0, Viseme.AA, 0.5)])
    assert r.current_viseme(0.5) is Viseme.SIL
    assert r.current_viseme(1.2) is Viseme.AA
    assert r.current_viseme(1.6) is Viseme.SIL
    r.append([VisemeEvent(2.0, Viseme.OO, 0.5)], [EmotionEvent(2.0, Emotion.SAD)])
    assert r.current_viseme(1.8) is Viseme.SIL
    assert r.current_viseme(2.1) is Viseme.OO
    assert r.current_emotion(1.9) is Emotion.NEUTRAL
    assert r.current_emotion(2.5) is Emotion.SAD
    assert r.end_time == pytest.approx(2.5)


def test_schedule_reader_rewind_and_clear():
    r = ScheduleReader([VisemeEvent(0.0, Viseme.AA, 1.0), VisemeEvent(1.0, Viseme.EE, 1.0)])
    assert r.current_viseme(1.5) is Viseme.EE
    assert r.current_viseme(0.5) is Viseme.AA      # clock went backwards
    r.clear()
    assert r.current_viseme(0.5) is Viseme.SIL


def test_schedule_reader_trim_keeps_active_emotion():
    r = ScheduleReader(
        [VisemeEvent(0.0, Viseme.AA, 1.0), VisemeEvent(5.0, Viseme.EE, 1.0)],
        [EmotionEvent(0.0, Emotion.HAPPY), EmotionEvent(10.0, Emotion.SAD)])
    r.trim_before(4.0)
    assert len(r.visemes) == 1
    assert r.current_emotion(6.0) is Emotion.HAPPY
    assert r.current_viseme(5.5) is Viseme.EE


def test_fit_word_times_spreads_words_across_a_known_duration():
    from talker.phoneme_scheduler import fit_word_times
    wt = fit_word_times("Hello there my friend", 2.0)
    assert [w for w, _, _ in wt] == ["Hello", "there", "my", "friend"]
    assert wt[0][1] >= 0.0 and wt[-1][2] <= 2.0
    assert all(wt[i][2] <= wt[i + 1][1] + 1e-6 for i in range(len(wt) - 1))
    assert wt[-1][2] > 1.5                                   # fills most of the clip
    assert fit_word_times("", 2.0) == [] and fit_word_times("hi", 0) == []


def test_emotion_settles_back_to_neutral_when_no_new_tag_arrives():
    """An expression holds until the next tag, then relaxes rather than sticking forever."""
    from talker.phoneme_scheduler import Emotion, EmotionEvent, ScheduleReader
    s = ScheduleReader(emotion_hold=45.0)
    s.append((), [EmotionEvent(10.0, Emotion.HAPPY)])
    assert s.current_emotion(9.0) is Emotion.NEUTRAL         # before the tag
    assert s.current_emotion(10.0) is Emotion.HAPPY
    assert s.current_emotion(54.0) is Emotion.HAPPY          # still inside the window
    assert s.current_emotion(56.0) is Emotion.NEUTRAL        # nothing new: settled back
    s.append((), [EmotionEvent(56.0, Emotion.ANGRY)])        # a new tag restarts the hold
    assert s.current_emotion(60.0) is Emotion.ANGRY
    assert s.current_emotion(102.0) is Emotion.NEUTRAL
    forever = ScheduleReader(emotion_hold=0)                 # 0 keeps the old behaviour
    forever.append((), [EmotionEvent(1.0, Emotion.SAD)])
    assert forever.current_emotion(9999.0) is Emotion.SAD
