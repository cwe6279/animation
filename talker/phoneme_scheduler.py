"""
talker/phoneme_scheduler.py
====================
Pure, incremental text -> phoneme -> viseme scheduling.

Everything in this module is synchronous and side-effect free, so it can be
unit tested without audio hardware, a display, or network access.

Pipeline (driven by speech_pipeline.SpeechPipeline):
  text with optional [emotion] tags
    -> split into sentences            (SentenceSplitter, streaming-friendly)
    -> strip emotion tags per sentence (parse_emotion_tags)
    -> TTS backend yields word timestamps as audio streams in
    -> word_to_viseme_events() per word (g2p-en, or a regex fallback)
    -> ScheduleReader.append() places the events on the shared audio timeline

The renderer asks ScheduleReader for the current viseme + emotion each frame.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple


# ─────────────────────────────────────────────────────
# VISEME ENUM  (Preston Blair's classic set, extended)
# ─────────────────────────────────────────────────────
class Viseme(Enum):
    SIL = "sil"   # closed / rest
    PP  = "pp"    # pressed lips: p b m
    FF  = "ff"    # lip-teeth: f v
    TH  = "th"    # tongue tip: th dh
    DD  = "dd"    # tongue up: t d n l
    KK  = "kk"    # back open: k g ng
    CH  = "ch"    # puckered forward: ch sh zh j
    SS  = "ss"    # sibilant: s z
    AA  = "aa"    # wide open: a aw ah ae
    EE  = "ee"    # wide flat: ee ih
    OO  = "oo"    # tight round: oo ow uh
    AH  = "ah"    # neutral open: uh schwa er


# ─────────────────────────────────────────────────────
# EMOTION ENUM  (inline tags like [angry] in text)
# ─────────────────────────────────────────────────────
class Emotion(Enum):
    NEUTRAL  = "neutral"
    HAPPY    = "happy"
    ANGRY    = "angry"
    ANNOYED  = "annoyed"
    SAD      = "sad"
    SURPRISE = "surprise"


EMOTION_NAMES = {e.value: e for e in Emotion}


def parse_emotion(name: Optional[str]) -> Optional[Emotion]:
    """Case-insensitive lookup; returns None for unknown names."""
    if not name:
        return None
    return EMOTION_NAMES.get(name.strip().lower())


@dataclass
class EmotionEvent:
    time: float        # seconds on the audio timeline
    emotion: Emotion


@dataclass
class VisemeEvent:
    time: float        # seconds on the audio timeline
    viseme: Viseme
    duration: float    # seconds this viseme lasts

    @property
    def end_time(self) -> float:
        return self.time + self.duration


# ─────────────────────────────────────────────────────
# VISEME PROPERTIES (for renderer)
# ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class VisemeProps:
    open_amount: float   # vertical opening 0..1
    width_scale: float   # horizontal width multiplier 0..1
    rounded: bool        # pursed/round vs flat
    label: str


VISEME_PROPS: dict[Viseme, VisemeProps] = {
    Viseme.SIL: VisemeProps(0.00, 0.80, False, "silence"),
    Viseme.PP:  VisemeProps(0.00, 0.70, False, "press"),
    Viseme.FF:  VisemeProps(0.15, 0.65, False, "lip-teeth"),
    Viseme.TH:  VisemeProps(0.20, 0.60, False, "tongue-tip"),
    Viseme.DD:  VisemeProps(0.25, 0.70, False, "tongue-up"),
    Viseme.KK:  VisemeProps(0.40, 0.75, False, "back-open"),
    Viseme.CH:  VisemeProps(0.35, 0.55, True,  "puckered"),
    Viseme.SS:  VisemeProps(0.18, 0.65, False, "sibilant"),
    Viseme.AA:  VisemeProps(0.85, 1.00, False, "wide-open"),
    Viseme.EE:  VisemeProps(0.45, 1.00, False, "wide-flat"),
    Viseme.OO:  VisemeProps(0.55, 0.50, True,  "round"),
    Viseme.AH:  VisemeProps(0.50, 0.80, False, "neutral"),
}


# ─────────────────────────────────────────────────────
# ARPABET -> VISEME MAP  (all 39 ARPAbet phonemes + silence)
# ─────────────────────────────────────────────────────
ARPABET_TO_VISEME: dict[str, Viseme] = {
    "SIL": Viseme.SIL, "SP": Viseme.SIL,
    # Stops
    "P": Viseme.PP, "B": Viseme.PP, "T": Viseme.DD, "D": Viseme.DD,
    "K": Viseme.KK, "G": Viseme.KK,
    # Fricatives
    "F": Viseme.FF, "V": Viseme.FF, "TH": Viseme.TH, "DH": Viseme.TH,
    "S": Viseme.SS, "Z": Viseme.SS, "SH": Viseme.CH, "ZH": Viseme.CH,
    "HH": Viseme.AH,
    # Affricates
    "CH": Viseme.CH, "JH": Viseme.CH,
    # Nasals
    "M": Viseme.PP, "N": Viseme.DD, "NG": Viseme.KK,
    # Liquids & glides
    "L": Viseme.DD, "R": Viseme.AH, "W": Viseme.OO, "Y": Viseme.EE,
    # Vowels (stress digits stripped before lookup)
    "AA": Viseme.AA, "AE": Viseme.AA, "AH": Viseme.AH, "AO": Viseme.AA,
    "AW": Viseme.OO, "AY": Viseme.AA, "EH": Viseme.EE, "ER": Viseme.AH,
    "EY": Viseme.EE, "IH": Viseme.EE, "IY": Viseme.EE, "OW": Viseme.OO,
    "OY": Viseme.OO, "UH": Viseme.OO, "UW": Viseme.OO,
}


# ─────────────────────────────────────────────────────
# TAG PARSING
# ─────────────────────────────────────────────────────
# Text carries [bracketed] performance tags in the ElevenLabs v3 style:
# emotional states, reactions, tone cues, character cues ("[sigh]",
# "[light chuckle]", "[British accent]"). Two consumers:
#   * the voice: backends that support audio tags get the text with tags kept;
#     others get the clean text (a plain TTS would read "sigh" aloud)
#   * the face: tags are mapped onto the six eye expressions below; tags with
#     no facial meaning (e.g. [whispers], [pauses]) leave the eyes alone.
_TAG_RE = re.compile(r"\[([^\[\]]{1,40})\]")

TAG_TO_EMOTION: dict[str, Emotion] = {}
for _emo, _words in {
    Emotion.HAPPY: ["happy", "excited", "cheerful", "cheerfully", "playful", "playfully", "laughs",
                    "laughing", "laugh", "giggle", "giggles", "giggling", "chuckle", "chuckles",
                    "chuckling", "joyful", "joy", "amused", "delighted", "warmly", "proud", "relieved",
                    "relief", "sigh of relief", "grinning", "smiling", "smiles", "gleeful", "thrilled", "fondly",
                    "teasing", "flirtatiously", "enthusiastic"],
    Emotion.ANGRY: ["angry", "furious", "frustrated", "enraged", "shouting", "shouts", "yelling",
                    "yells", "growls", "growling", "snaps", "snarls", "outraged", "mad", "livid",
                    "fuming", "hostile", "threatening", "menacing"],
    Emotion.ANNOYED: ["annoyed", "irritated", "sarcastically", "sarcastic", "whiny", "grumbles",
                      "grumbling", "impatient", "impatiently", "exasperated", "bored", "unimpressed",
                      "sighs heavily", "scoffs", "smug", "condescending", "dry", "dryly", "deadpan"],
    Emotion.SAD: ["sad", "sorrowful", "sorrow", "tired", "exhausted", "regretful", "regret", "resigned",
                  "hesitant", "hesitates", "hesitantly", "disappointed", "mournful", "crying", "sobbing",
                  "sobs", "gloomy", "melancholy", "hurt", "apologetic", "wistful", "lonely",
                  "heartbroken", "weary", "defeated", "somber", "sombre", "sigh", "sighs", "sighing",
                  "quietly", "softly"],
    Emotion.SURPRISE: ["surprise", "surprised", "awe", "amazed", "gasps", "gasp", "gasping", "shocked",
                       "astonished", "startled", "stunned", "wow", "curious", "intrigued", "alarmed",
                       "scared", "afraid", "terrified", "fearful", "nervous", "nervously", "anxious",
                       "panicked", "wide-eyed", "bewildered", "confused", "puzzled"],
    Emotion.NEUTRAL: ["neutral", "calm", "calmly", "matter-of-fact", "flatly", "flat", "serious",
                      "thoughtful", "thoughtfully", "composed", "sternly", "firmly", "gently",
                      "reassuring", "confident", "confidently", "whispers", "whispering", "whisper"],
}.items():
    for _w in _words:
        TAG_TO_EMOTION[_w] = _emo


def tag_to_emotion(tag: str) -> Optional[Emotion]:
    """
    Map a performance tag to a face emotion, or None if it says nothing about
    the eyes ("[pauses]", "[British accent]"). Tries the whole tag, then its
    words, then simple suffix trims ("nervously" -> "nervous", "sad tone" -> "sad").
    """
    t = tag.strip().lower()
    if t in TAG_TO_EMOTION:
        return TAG_TO_EMOTION[t]
    words = [w for w in re.split(r"[\s,/-]+", t) if w and w not in ("tone", "voice", "of", "a", "with")]
    for w in words:
        if w in TAG_TO_EMOTION:
            return TAG_TO_EMOTION[w]
    for w in words:
        for suffix in ("ly", "ed", "ing", "s"):
            if w.endswith(suffix) and w[: -len(suffix)] in TAG_TO_EMOTION:
                return TAG_TO_EMOTION[w[: -len(suffix)]]
    return None


def parse_tags(text: str) -> Tuple[str, str, List[Tuple[int, Emotion]]]:
    """
    Returns (clean_text, voice_text, [(word_index, emotion), ...]).

      clean_text  tags removed (for TTS that cannot perform them, and for
                  counting words)
      voice_text  tags kept, normalized to one space around them (for v3)
      emotions    face emotion changes keyed by the index of the clean-text
                  word they precede; only tags with a facial meaning appear

        "[angry] I'm so mad! [sigh] But also hurt."
        -> ("I'm so mad! But also hurt.",
            "[angry] I'm so mad! [sigh] But also hurt.",
            [(0, ANGRY), (3, SAD)])
    """
    tags: List[Tuple[int, Emotion]] = []
    clean_parts: List[str] = []
    word_count = 0
    pos = 0
    for match in _TAG_RE.finditer(text):
        before = text[pos:match.start()]
        if before.strip():
            clean_parts.append(before)
            word_count += len(before.split())
        emotion = tag_to_emotion(match.group(1))
        if emotion is not None:
            if tags and tags[-1][0] == word_count:
                tags[-1] = (word_count, emotion)     # stacked tags: last one wins
            else:
                tags.append((word_count, emotion))
        pos = match.end()
    remainder = text[pos:]
    if remainder.strip():
        clean_parts.append(remainder)
    clean_text = re.sub(r"\s{2,}", " ", "".join(clean_parts)).strip()
    voice_text = re.sub(r"\s{2,}", " ", _TAG_RE.sub(lambda m: f" [{m.group(1).strip()}] ", text)).strip()
    return clean_text, voice_text, tags


def parse_emotion_tags(text: str) -> Tuple[str, List[Tuple[int, Emotion]]]:
    """Backward-compatible wrapper: (clean_text, emotions)."""
    clean, _, tags = parse_tags(text)
    return clean, tags


def strip_tags(text: str) -> str:
    return parse_tags(text)[0]


# ─────────────────────────────────────────────────────
# SENTENCE SPLITTING  (batch and streaming)
# ─────────────────────────────────────────────────────
_SENTENCE_END_RE = re.compile(r"([.!?…]+[\"')\]]*)(\s+)")


class SentenceSplitter:
    """
    Incrementally splits text into sentence-sized chunks so TTS can start on
    the first sentence while the rest (e.g. an LLM response) is still arriving.

        s = SentenceSplitter()
        for chunk in llm_tokens:
            for sentence in s.feed(chunk): ...
        for sentence in s.flush(): ...

    Emotion tags stay attached to the words that follow them because they sit
    before those words in the text. Very short fragments are merged with the
    next sentence so we don't fire a TTS request for "Oh." alone.
    """

    def __init__(self, min_chars: int = 12):
        self.min_chars = min_chars
        self._buf = ""
        self._pending = ""   # short fragment waiting to be merged

    def feed(self, text: str) -> List[str]:
        self._buf += text
        out: List[str] = []
        while True:
            m = _SENTENCE_END_RE.search(self._buf)
            if not m:
                break
            sentence = self._buf[:m.end(1)]
            self._buf = self._buf[m.end():]
            out.extend(self._emit(sentence))
        return out

    def flush(self) -> List[str]:
        out: List[str] = []
        tail = (self._pending + " " + self._buf).strip()
        self._pending = ""
        self._buf = ""
        if tail:
            out.append(tail)
        return out

    def _emit(self, sentence: str) -> List[str]:
        sentence = (self._pending + " " + sentence).strip() if self._pending else sentence.strip()
        self._pending = ""
        if not sentence:
            return []
        # Count only real words, not tags, when deciding if it's too short.
        clean = strip_tags(sentence)
        if len(clean) < self.min_chars:
            self._pending = sentence
            return []
        return [sentence]


def split_sentences(text: str, min_chars: int = 12) -> List[str]:
    """Batch helper: split a whole string into sentence chunks."""
    s = SentenceSplitter(min_chars=min_chars)
    out = s.feed(text)
    out.extend(s.flush())
    return out


# ─────────────────────────────────────────────────────
# GRAPHEME -> ARPABET
# ─────────────────────────────────────────────────────
_GRAPHEME_RULES: List[Tuple[re.Pattern, List[str]]] = [
    # Digraphs & special combos first (order matters)
    (re.compile(r"ch"), ["CH"]),
    (re.compile(r"sh"), ["SH"]),
    (re.compile(r"th"), ["TH"]),
    (re.compile(r"wh"), ["W"]),
    (re.compile(r"ph"), ["F"]),
    (re.compile(r"ng"), ["NG"]),
    (re.compile(r"ck"), ["K"]),
    (re.compile(r"qu"), ["K", "W"]),
    (re.compile(r"gh"), []),           # silent gh
    # Vowel groups
    (re.compile(r"ee|ea"), ["IY"]),
    (re.compile(r"oo"), ["UW"]),
    (re.compile(r"ou|ow"), ["AW"]),
    (re.compile(r"oi|oy"), ["OY"]),
    (re.compile(r"ai|ay"), ["EY"]),
    (re.compile(r"au|aw"), ["AO"]),
    # Single consonants
    (re.compile(r"b"), ["B"]),
    (re.compile(r"c(?=[ei])"), ["S"]),
    (re.compile(r"c"), ["K"]),
    (re.compile(r"d"), ["D"]),
    (re.compile(r"f"), ["F"]),
    (re.compile(r"g(?=[ei])"), ["JH"]),
    (re.compile(r"g"), ["G"]),
    (re.compile(r"h"), ["HH"]),
    (re.compile(r"j"), ["JH"]),
    (re.compile(r"k"), ["K"]),
    (re.compile(r"l"), ["L"]),
    (re.compile(r"m"), ["M"]),
    (re.compile(r"n"), ["N"]),
    (re.compile(r"p"), ["P"]),
    (re.compile(r"r"), ["R"]),
    (re.compile(r"s"), ["S"]),
    (re.compile(r"t"), ["T"]),
    (re.compile(r"v"), ["V"]),
    (re.compile(r"w"), ["W"]),
    (re.compile(r"x"), ["K", "S"]),
    (re.compile(r"y"), ["Y"]),
    (re.compile(r"z"), ["Z"]),
    # Single vowels (catch-all)
    (re.compile(r"a"), ["AE"]),
    (re.compile(r"e"), ["EH"]),
    (re.compile(r"i"), ["IH"]),
    (re.compile(r"o"), ["OW"]),
    (re.compile(r"u"), ["AH"]),
]

_STRESS_RE = re.compile(r"\d")


def grapheme_to_arpabet_fallback(word: str) -> List[str]:
    """Regex-based grapheme -> ARPAbet approximation (~85% on common words)."""
    phonemes: List[str] = []
    text = word.lower().strip(".,!?;:\"'")
    i = 0
    while i < len(text):
        for pattern, phones in _GRAPHEME_RULES:
            m = pattern.match(text, i)
            if m:
                phonemes.extend(phones)
                i = m.end()
                break
        else:
            i += 1  # skip unknown char
    return phonemes


# g2p-en singleton. Import + model load takes 1-2 s the first time, so the
# app warms it up in a background thread at startup (see warm_up_g2p).
_g2p_lock = threading.Lock()
_g2p_instance = None
_g2p_unavailable = False


def _ensure_nltk_data() -> None:
    """g2p-en needs two NLTK corpora; fetch them once if missing (network)."""
    import nltk
    for res, pkg in (("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
                     ("corpora/cmudict", "cmudict")):
        try:
            nltk.data.find(res)
        except LookupError:
            print(f"[g2p] downloading NLTK data: {pkg}")
            nltk.download(pkg, quiet=True)


def _get_g2p():
    global _g2p_instance, _g2p_unavailable
    if _g2p_instance is not None or _g2p_unavailable:
        return _g2p_instance
    with _g2p_lock:
        if _g2p_instance is None and not _g2p_unavailable:
            try:
                _ensure_nltk_data()
                from g2p_en import G2p
                inst = G2p()
                inst("warm up")
                _g2p_instance = inst
            except Exception as e:  # ImportError, missing NLTK data, etc.
                _g2p_unavailable = True
                print(f"[g2p] g2p-en unavailable ({e.__class__.__name__}: "
                      f"{str(e).strip().splitlines()[0] if str(e).strip() else ''}); "
                      "using regex fallback")
    return _g2p_instance


def warm_up_g2p() -> None:
    """Load g2p-en now (blocking). Call from a background thread at startup."""
    _get_g2p()


def word_to_arpabet(word: str) -> List[str]:
    """Convert a word to ARPAbet phonemes (stress digits stripped)."""
    g2p = _get_g2p()
    if g2p is not None:
        try:
            raw = g2p(word)
            return [_STRESS_RE.sub("", p) for p in raw if p.strip()]
        except Exception:
            pass
    return grapheme_to_arpabet_fallback(word)


def arpabet_to_visemes(phonemes: Iterable[str]) -> List[Viseme]:
    return [ARPABET_TO_VISEME.get(_STRESS_RE.sub("", p).upper(), Viseme.AH)
            for p in phonemes]


# ─────────────────────────────────────────────────────
# WORD -> TIMED VISEME EVENTS
# ─────────────────────────────────────────────────────
MIN_WORD_SECONDS = 0.05


def word_to_viseme_events(word: str, t_start: float, t_end: float) -> List[VisemeEvent]:
    """
    Spread a word's visemes across [t_start, t_end]. Consonants get less time
    than vowels, and consecutive duplicate shapes are merged so the mouth
    doesn't re-hit the same pose.
    """
    duration = max(t_end - t_start, MIN_WORD_SECONDS)
    visemes = arpabet_to_visemes(word_to_arpabet(word))
    if not visemes:
        return [VisemeEvent(t_start, Viseme.SIL, duration)]

    deduped = [visemes[0]]
    for v in visemes[1:]:
        if v != deduped[-1]:
            deduped.append(v)

    weights = [0.6 + VISEME_PROPS[v].open_amount * 0.8 for v in deduped]
    total_w = sum(weights)
    events: List[VisemeEvent] = []
    t = t_start
    for v, w in zip(deduped, weights):
        dur = duration * (w / total_w)
        events.append(VisemeEvent(t, v, dur))
        t += dur
    return events


def estimate_word_times(text: str, phoneme_seconds: float = 0.08,
                        gap_seconds: float = 0.06) -> List[Tuple[str, float, float]]:
    """
    No timestamps available: estimate word timing from phoneme counts.
    Used by backends that can't report word boundaries.
    """
    out: List[Tuple[str, float, float]] = []
    t = 0.0
    for word in re.findall(r"[A-Za-z']+", text):
        n = max(1, len(word_to_arpabet(word)))
        dur = n * phoneme_seconds
        out.append((word, t, t + dur))
        t += dur + gap_seconds
    return out


# ─────────────────────────────────────────────────────
# SCHEDULE READER  (queried each frame by the renderer)
# ─────────────────────────────────────────────────────
class ScheduleReader:
    """
    A growing, time-sorted list of viseme and emotion events on the shared
    audio timeline. The speech pipeline appends from a background thread
    while the render loop reads each frame; O(1) amortized per query.

    Events must be appended in non-decreasing time order per utterance, and
    utterances are placed on the timeline in playback order, so the list
    stays sorted by construction.
    """

    def __init__(self, visemes: Optional[List[VisemeEvent]] = None,
                 emotions: Optional[List[EmotionEvent]] = None):
        self._lock = threading.Lock()
        self.visemes: List[VisemeEvent] = list(visemes or [])
        self.emotions: List[EmotionEvent] = list(emotions or [])
        self._vidx = 0
        self._eidx = 0

    # -- writers -------------------------------------------------------
    def append(self, visemes: Iterable[VisemeEvent] = (),
               emotions: Iterable[EmotionEvent] = ()) -> None:
        with self._lock:
            self.visemes.extend(visemes)
            self.emotions.extend(emotions)

    def clear(self) -> None:
        with self._lock:
            self.visemes.clear()
            self.emotions.clear()
            self._vidx = 0
            self._eidx = 0

    def trim_before(self, t: float) -> None:
        """Drop events that ended before time t (keeps memory flat in long sessions)."""
        with self._lock:
            keep = 0
            while keep < len(self.visemes) and self.visemes[keep].end_time < t:
                keep += 1
            if keep:
                del self.visemes[:keep]
                self._vidx = max(0, self._vidx - keep)
            ekeep = 0
            # Keep the last emotion at or before t so it stays in effect.
            while ekeep + 1 < len(self.emotions) and self.emotions[ekeep + 1].time <= t:
                ekeep += 1
            if ekeep:
                del self.emotions[:ekeep]
                self._eidx = max(0, self._eidx - ekeep)

    @property
    def end_time(self) -> float:
        with self._lock:
            return self.visemes[-1].end_time if self.visemes else 0.0

    # -- readers -------------------------------------------------------
    def current_viseme(self, t: float) -> Viseme:
        with self._lock:
            sched = self.visemes
            if not sched:
                return Viseme.SIL
            # Clock went backwards (timeline reset)? Rescan from the start.
            if self._vidx >= len(sched) or sched[self._vidx].time > t and self._vidx > 0:
                self._vidx = 0
            while self._vidx < len(sched) - 1 and sched[self._vidx].end_time <= t:
                self._vidx += 1
            ev = sched[self._vidx]
            if t < ev.time or t >= ev.end_time:
                return Viseme.SIL   # in a gap, or past the last event
            return ev.viseme

    def current_emotion(self, t: float) -> Emotion:
        with self._lock:
            evs = self.emotions
            if not evs:
                return Emotion.NEUTRAL
            if self._eidx >= len(evs) or evs[self._eidx].time > t and self._eidx > 0:
                self._eidx = 0
            while self._eidx < len(evs) - 1 and evs[self._eidx + 1].time <= t:
                self._eidx += 1
            ev = evs[self._eidx]
            return ev.emotion if t >= ev.time else Emotion.NEUTRAL
