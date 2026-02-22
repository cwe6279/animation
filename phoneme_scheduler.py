"""
phoneme_scheduler.py
Converts text → phonemes → timed viseme schedule

Pipeline:
  text (with optional [emotion] tags)
    → strip emotion tags, record positions
    → edge-tts word timestamps  (word start/end times in seconds)
    → g2p-en phoneme strings    (per word)
    → viseme codes              (per phoneme)
    → spread across word duration
    → sorted list of VisemeEvent(time, viseme, duration)
    → sorted list of EmotionEvent(time, emotion)

Fallback: if g2p-en not installed, uses a fast regex-based
          English approximation that covers ~85% of common words.
"""

from __future__ import annotations
import re
import asyncio
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────────────
# VISEME ENUM  (matches Preston Blair's classic 10-shape set
#               extended with a few extras for clarity)
# ─────────────────────────────────────────────────────
class Viseme(Enum):
    # Shape name      mouth description
    SIL   = "sil"   # closed / rest
    PP    = "pp"    # pressed lips: p b m
    FF    = "ff"    # lip-teeth: f v
    TH    = "th"    # tongue tip: th ð
    DD    = "dd"    # tongue up: t d n l
    KK    = "kk"    # back open: k g ng
    CH    = "ch"    # puckered forward: ch sh zh j
    SS    = "ss"    # sibilant: s z
    AA    = "aa"    # wide open: a aw ah æ
    EE    = "ee"    # wide flat: ee ih
    OO    = "oo"    # tight round: oo ow uh
    AH    = "ah"    # neutral open: uh schwa er


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

_EMOTION_NAMES = {e.value: e for e in Emotion}


@dataclass
class EmotionEvent:
    time: float        # seconds from audio start
    emotion: Emotion


def _parse_emotion_tags(text: str) -> Tuple[str, List[Tuple[int, Emotion]]]:
    """
    Strip [emotion] tags from text and record which word they precede.

    Example:
        "[angry]I'm so mad! [sad]But also hurt."
        → ("I'm so mad! But also hurt.",
           [(0, Emotion.ANGRY), (4, Emotion.SAD)])

    Returns (clean_text, [(word_index, emotion), ...])
    """
    tag_pattern = re.compile(r'\[(\w+)\]')
    tags: List[Tuple[int, Emotion]] = []
    clean_parts = []
    word_count = 0

    pos = 0
    for match in tag_pattern.finditer(text):
        # Text before this tag — count its words
        before = text[pos:match.start()]
        if before.strip():
            words_in_chunk = before.split()
            clean_parts.append(before)
            word_count += len(words_in_chunk)

        tag_name = match.group(1).lower()
        if tag_name in _EMOTION_NAMES:
            tags.append((word_count, _EMOTION_NAMES[tag_name]))
        pos = match.end()

    # Remaining text after last tag
    remainder = text[pos:]
    if remainder:
        clean_parts.append(remainder)

    clean_text = "".join(clean_parts).strip()
    # Collapse extra spaces from tag removal
    clean_text = re.sub(r'  +', ' ', clean_text)
    return clean_text, tags


# ─────────────────────────────────────────────────────
# ARPABET → VISEME MAP
# Covers all 39 ARPAbet phonemes used by g2p-en
# ─────────────────────────────────────────────────────
ARPABET_TO_VISEME: dict[str, Viseme] = {
    # Silence
    "SIL": Viseme.SIL,
    "SP":  Viseme.SIL,

    # Stops
    "P":  Viseme.PP,
    "B":  Viseme.PP,
    "T":  Viseme.DD,
    "D":  Viseme.DD,
    "K":  Viseme.KK,
    "G":  Viseme.KK,

    # Fricatives
    "F":  Viseme.FF,
    "V":  Viseme.FF,
    "TH": Viseme.TH,
    "DH": Viseme.TH,
    "S":  Viseme.SS,
    "Z":  Viseme.SS,
    "SH": Viseme.CH,
    "ZH": Viseme.CH,
    "HH": Viseme.AH,

    # Affricates
    "CH": Viseme.CH,
    "JH": Viseme.CH,

    # Nasals
    "M":  Viseme.PP,
    "N":  Viseme.DD,
    "NG": Viseme.KK,

    # Liquids & glides
    "L":  Viseme.DD,
    "R":  Viseme.AH,
    "W":  Viseme.OO,
    "Y":  Viseme.EE,

    # Vowels — stress markers stripped before lookup
    "AA": Viseme.AA,
    "AE": Viseme.AA,
    "AH": Viseme.AH,
    "AO": Viseme.AA,
    "AW": Viseme.OO,
    "AY": Viseme.AA,
    "EH": Viseme.EE,
    "ER": Viseme.AH,
    "EY": Viseme.EE,
    "IH": Viseme.EE,
    "IY": Viseme.EE,
    "OW": Viseme.OO,
    "OY": Viseme.OO,
    "UH": Viseme.OO,
    "UW": Viseme.OO,
}


# ─────────────────────────────────────────────────────
# VISEME PROPERTIES (for renderer)
# open_amount: 0..1  width_scale: 0..1  shape: round|wide|neutral
# ─────────────────────────────────────────────────────
@dataclass
class VisemeProps:
    open_amount: float   # vertical opening
    width_scale: float   # horizontal width multiplier
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
# TIMED VISEME EVENT
# ─────────────────────────────────────────────────────
@dataclass
class VisemeEvent:
    time: float      # seconds from audio start
    viseme: Viseme
    duration: float  # seconds this viseme lasts

    def end_time(self) -> float:
        return self.time + self.duration


# ─────────────────────────────────────────────────────
# FALLBACK: REGEX PHONEME APPROXIMATION
# Fast pure-Python grapheme-to-phoneme for common English
# patterns. No external deps. ~85% accuracy on common words.
# ─────────────────────────────────────────────────────

_GRAPHEME_RULES: List[Tuple[re.Pattern, List[str]]] = [
    # Digraphs & special combos first (order matters)
    (re.compile(r'ch',   re.I), ["CH"]),
    (re.compile(r'sh',   re.I), ["SH"]),
    (re.compile(r'th',   re.I), ["TH"]),
    (re.compile(r'wh',   re.I), ["W"]),
    (re.compile(r'ph',   re.I), ["F"]),
    (re.compile(r'ng',   re.I), ["NG"]),
    (re.compile(r'ck',   re.I), ["K"]),
    (re.compile(r'qu',   re.I), ["K", "W"]),
    (re.compile(r'gh',   re.I), []),           # silent gh
    # Vowel groups
    (re.compile(r'ee|ea', re.I), ["IY"]),
    (re.compile(r'oo',   re.I), ["UW"]),
    (re.compile(r'ou|ow', re.I), ["AW"]),
    (re.compile(r'oi|oy', re.I), ["OY"]),
    (re.compile(r'ai|ay', re.I), ["EY"]),
    (re.compile(r'au|aw', re.I), ["AO"]),
    # Single consonants
    (re.compile(r'b', re.I), ["B"]),
    (re.compile(r'c(?=[ei])', re.I), ["S"]),
    (re.compile(r'c', re.I), ["K"]),
    (re.compile(r'd', re.I), ["D"]),
    (re.compile(r'f', re.I), ["F"]),
    (re.compile(r'g(?=[ei])', re.I), ["JH"]),
    (re.compile(r'g', re.I), ["G"]),
    (re.compile(r'h', re.I), ["HH"]),
    (re.compile(r'j', re.I), ["JH"]),
    (re.compile(r'k', re.I), ["K"]),
    (re.compile(r'l', re.I), ["L"]),
    (re.compile(r'm', re.I), ["M"]),
    (re.compile(r'n', re.I), ["N"]),
    (re.compile(r'p', re.I), ["P"]),
    (re.compile(r'r', re.I), ["R"]),
    (re.compile(r's', re.I), ["S"]),
    (re.compile(r't', re.I), ["T"]),
    (re.compile(r'v', re.I), ["V"]),
    (re.compile(r'w', re.I), ["W"]),
    (re.compile(r'x', re.I), ["K", "S"]),
    (re.compile(r'y', re.I), ["Y"]),
    (re.compile(r'z', re.I), ["Z"]),
    # Single vowels (catch-all)
    (re.compile(r'a', re.I), ["AE"]),
    (re.compile(r'e', re.I), ["EH"]),
    (re.compile(r'i', re.I), ["IH"]),
    (re.compile(r'o', re.I), ["OW"]),
    (re.compile(r'u', re.I), ["AH"]),
]

def _grapheme_to_arpabet_fallback(word: str) -> List[str]:
    """Regex-based grapheme → ARPAbet approximation."""
    phonemes = []
    text = word.lower().strip(".,!?;:\"'")
    i = 0
    while i < len(text):
        matched = False
        for pattern, phones in _GRAPHEME_RULES:
            m = pattern.match(text, i)
            if m:
                phonemes.extend(phones)
                i = m.end()
                matched = True
                break
        if not matched:
            i += 1  # skip unknown char
    return phonemes


def _word_to_arpabet(word: str) -> List[str]:
    """Convert word to ARPAbet phonemes. Tries g2p-en first."""
    try:
        from g2p_en import G2p
        if not hasattr(_word_to_arpabet, '_g2p'):
            _word_to_arpabet._g2p = G2p()
        raw = _word_to_arpabet._g2p(word)
        # Strip stress markers (0,1,2 digits)
        return [re.sub(r'\d', '', p) for p in raw if p.strip() and p != ' ']
    except Exception:
        # ImportError (g2p not installed) or runtime error (NLTK data missing, etc.)
        return _grapheme_to_arpabet_fallback(word)


def _arpabet_to_visemes(phonemes: List[str]) -> List[Viseme]:
    result = []
    for p in phonemes:
        p_clean = re.sub(r'\d', '', p).upper()
        v = ARPABET_TO_VISEME.get(p_clean, Viseme.AH)
        result.append(v)
    return result


# ─────────────────────────────────────────────────────
# MP3 → WAV CONVERSION
# Tries: 1) system ffmpeg  2) imageio-ffmpeg bundle  3) pydub
# ─────────────────────────────────────────────────────
def _find_ffmpeg() -> str | None:
    """Return path to ffmpeg binary, or None."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _convert_mp3_to_wav(mp3_path: str, wav_path: str):
    """Convert MP3 to 22050 Hz mono WAV for pyaudio playback."""
    import subprocess
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-y", "-i", mp3_path, "-ar", "22050", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True
        )
        return
    # Fallback: try pydub (needs its own ffmpeg config)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(mp3_path)
        seg = seg.set_frame_rate(22050).set_channels(1)
        seg.export(wav_path, format="wav")
        return
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found. Install it:\n"
        "  pip install imageio-ffmpeg\n"
        "  OR download from https://ffmpeg.org and add to PATH"
    )


# ─────────────────────────────────────────────────────
# EDGE-TTS WORD TIMESTAMP EXTRACTION
# Returns list of (word, start_sec, end_sec)
# ─────────────────────────────────────────────────────
async def _edge_tts_with_timestamps(
    text: str,
    voice: str,
    wav_path: str
) -> List[Tuple[str, float, float]]:
    """
    Run edge-tts, save WAV, collect word boundary events.
    Returns [(word, start_s, end_s), ...]
    """
    import edge_tts

    word_events = []
    communicate = edge_tts.Communicate(text, voice=voice, boundary="WordBoundary")

    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            # offset is in 100-nanosecond units
            start_s = chunk["offset"]  / 1e7
            dur_s   = chunk["duration"] / 1e7
            word_events.append((chunk["text"], start_s, start_s + dur_s))

    # Write raw audio (edge-tts gives MP3 by default)
    raw_path = wav_path.replace(".wav", ".mp3")
    with open(raw_path, "wb") as f:
        for chunk in audio_chunks:
            f.write(chunk)

    # Convert MP3 → WAV (needed for pyaudio playback)
    _convert_mp3_to_wav(raw_path, wav_path)

    print(f"[scheduler] got {len(word_events)} word timestamps from edge-tts")
    return word_events


# ─────────────────────────────────────────────────────
# MAIN SCHEDULER
# ─────────────────────────────────────────────────────
class PhonemeScheduler:
    """
    Converts text → (wav_path, List[VisemeEvent], List[EmotionEvent])
    Ready to hand off to the renderer.

    Supports inline emotion tags: "[angry]I'm mad! [sad]But also hurt."
    Tags are stripped before TTS and mapped to timed EmotionEvents.
    """

    # Typical phoneme duration in ms (used when no TTS timestamps)
    PHONEME_MS = 80

    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice

    def build(
        self,
        text: str,
        wav_path: str = None
    ) -> Tuple[str, List[VisemeEvent], List[EmotionEvent]]:
        """
        Synchronous entry point.
        Returns (wav_path, viseme_schedule, emotion_events) ready for playback.
        """
        # Parse and strip emotion tags before sending to TTS
        clean_text, emotion_tags = _parse_emotion_tags(text)

        if wav_path is None:
            wav_path = os.path.join(tempfile.gettempdir(), "talker_tts.wav")
        word_times = None
        try:
            import edge_tts
            word_times = asyncio.run(
                _edge_tts_with_timestamps(clean_text, self.voice, wav_path)
            )
            schedule = self._build_schedule_from_word_times(word_times)
        except ImportError:
            print("[warn] edge-tts not found — using pyttsx3 + estimated timing")
            wav_path = self._tts_pyttsx3(clean_text, wav_path)
            schedule = self._build_schedule_estimated(clean_text)

        # Map emotion tag word positions to timestamps
        emotion_events = self._build_emotion_events(emotion_tags, word_times, clean_text)
        if emotion_events:
            print(f"[scheduler] {len(emotion_events)} emotion events: "
                  + ", ".join(f"{e.emotion.value}@{e.time:.2f}s" for e in emotion_events))

        return wav_path, schedule, emotion_events

    def _build_emotion_events(
        self,
        tags: List[Tuple[int, Emotion]],
        word_times: Optional[List[Tuple[str, float, float]]],
        clean_text: str
    ) -> List[EmotionEvent]:
        """Map emotion tag positions (word indices) to timestamps."""
        if not tags:
            return []

        events = []
        for word_idx, emotion in tags:
            if word_times and word_idx < len(word_times):
                # Use the start time of the word this tag precedes
                t = word_times[word_idx][1]
            elif word_times and word_idx >= len(word_times):
                # Tag after all words — use end of last word
                t = word_times[-1][2] if word_times else 0.0
            else:
                # No word timestamps (pyttsx3 fallback) — estimate from word index
                words = clean_text.split()
                avg_word_dur = 0.4  # rough estimate
                t = word_idx * avg_word_dur
            events.append(EmotionEvent(time=t, emotion=emotion))

        return events

    def _build_schedule_from_word_times(
        self,
        word_times: List[Tuple[str, float, float]]
    ) -> List[VisemeEvent]:
        """
        Spread phoneme visemes evenly across each word's time window.
        Adds small leading SIL gaps between words.
        """
        events: List[VisemeEvent] = []

        for word, t_start, t_end in word_times:
            duration = max(t_end - t_start, 0.05)
            phonemes = _word_to_arpabet(word)
            if not phonemes:
                events.append(VisemeEvent(t_start, Viseme.SIL, duration))
                continue

            visemes = _arpabet_to_visemes(phonemes)

            # Remove consecutive duplicates — mouth doesn't re-hit same shape
            deduped = [visemes[0]]
            for v in visemes[1:]:
                if v != deduped[-1]:
                    deduped.append(v)
            visemes = deduped

            # Weight durations: consonants shorter, vowels longer
            weights = []
            for v in visemes:
                props = VISEME_PROPS[v]
                weights.append(0.6 + props.open_amount * 0.8)
            total_w = sum(weights)

            t = t_start
            for v, w in zip(visemes, weights):
                dur = duration * (w / total_w)
                events.append(VisemeEvent(t, v, dur))
                t += dur

        events.sort(key=lambda e: e.time)
        return events

    def _build_schedule_estimated(self, text: str) -> List[VisemeEvent]:
        """
        No TTS timestamps available — estimate timing from
        average phoneme durations (~80ms each).
        """
        events: List[VisemeEvent] = []
        t = 0.0
        words = re.findall(r"[a-zA-Z']+", text)
        for word in words:
            phonemes = _word_to_arpabet(word)
            visemes  = _arpabet_to_visemes(phonemes)
            for v in visemes:
                dur = self.PHONEME_MS / 1000.0
                events.append(VisemeEvent(t, v, dur))
                t += dur
            # inter-word gap
            events.append(VisemeEvent(t, Viseme.SIL, 0.06))
            t += 0.06
        return events

    def _tts_pyttsx3(self, text: str, wav_path: str) -> str:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
        except Exception as e:
            print(f"[error] pyttsx3 TTS failed: {e}")
        return wav_path


# ─────────────────────────────────────────────────────
# SCHEDULE READER  (called each frame by renderer)
# ─────────────────────────────────────────────────────
class ScheduleReader:
    """
    Given a sorted list of VisemeEvents (and optional EmotionEvents)
    and a playback clock, returns the current Viseme and Emotion
    each frame. O(1) amortized via index tracking.
    """
    def __init__(self, schedule: List[VisemeEvent],
                 emotion_events: Optional[List[EmotionEvent]] = None):
        self.schedule = schedule
        self._idx = 0
        self.emotion_events = emotion_events or []
        self._emo_idx = 0

    def reset(self):
        self._idx = 0
        self._emo_idx = 0

    def current_viseme(self, playback_time: float) -> Viseme:
        """Call each frame with current audio playback time in seconds."""
        if not self.schedule:
            return Viseme.SIL

        # Advance index to keep up with playback time
        while (self._idx < len(self.schedule) - 1 and
               self.schedule[self._idx].end_time() <= playback_time):
            self._idx += 1

        event = self.schedule[self._idx]
        if playback_time < event.time:
            return Viseme.SIL  # gap before first event
        if playback_time >= event.end_time() and self._idx >= len(self.schedule) - 1:
            return Viseme.SIL  # past the last event — return to neutral

        return event.viseme

    def current_emotion(self, playback_time: float) -> Emotion:
        """Return the most recent emotion at this playback time."""
        if not self.emotion_events:
            return Emotion.NEUTRAL

        # Advance index to the latest emotion event at or before playback_time
        while (self._emo_idx < len(self.emotion_events) - 1 and
               self.emotion_events[self._emo_idx + 1].time <= playback_time):
            self._emo_idx += 1

        event = self.emotion_events[self._emo_idx]
        if playback_time < event.time:
            return Emotion.NEUTRAL  # before first emotion tag
        return event.emotion
