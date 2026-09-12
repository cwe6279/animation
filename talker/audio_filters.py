"""
audio_filters.py — small IIR filters for the microphone path.

A room is mostly noise below speech: mains hum, air conditioning, traffic, the
thud of someone setting a cup down, and the bottom end of the character's own
voice coming back off the speaker. None of it carries words, but all of it adds
energy, and the energy is what the speech gate and the barge-in detector read.
A high-pass removes it before anything measures the level.

The low-pass is not a preference, it is a correctness fix: the microphone often
opens at 48 kHz and the loop decimates to the 16 kHz the recognizer wants by
plain interpolation. Anything above 8 kHz folds back into the speech band as
alias noise unless it is filtered out first.

Both are one biquad, direct form II transposed, carrying state between calls so
a filter applied chunk by chunk in an audio callback matches the same filter
applied to the whole recording: no clicks at the chunk boundaries.

    hp = Biquad.high_pass(90.0, 16000)
    pcm = hp.process_bytes(pcm)          # int16 bytes in, int16 bytes out

No scipy: the coefficients are three lines of algebra and the inner loop is
numpy's lfilter-equivalent written out, which is fast enough for 64 ms chunks.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Butterworth Q for a single biquad: -3 dB at the cutoff, maximally flat.
BUTTERWORTH_Q = math.sqrt(0.5)


class Biquad:
    """One second-order section, 12 dB per octave, with persistent state."""

    __slots__ = ("b0", "b1", "b2", "a1", "a2", "_z1", "_z2", "cutoff", "kind")

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float,
                 cutoff: float = 0.0, kind: str = ""):
        self.b0, self.b1, self.b2, self.a1, self.a2 = b0, b1, b2, a1, a2
        self.cutoff, self.kind = cutoff, kind
        self._z1 = 0.0
        self._z2 = 0.0

    # ── design ───────────────────────────────────────────────────────────
    @staticmethod
    def _omega(cutoff: float, rate: int):
        # Guard the Nyquist edge: a cutoff at or above it has no meaning.
        cutoff = max(1.0, min(float(cutoff), rate * 0.49))
        w = 2.0 * math.pi * cutoff / rate
        alpha = math.sin(w) / (2.0 * BUTTERWORTH_Q)
        return w, alpha, cutoff

    @classmethod
    def high_pass(cls, cutoff: float, rate: int) -> "Biquad":
        w, alpha, cutoff = cls._omega(cutoff, rate)
        cos_w = math.cos(w)
        a0 = 1.0 + alpha
        return cls((1.0 + cos_w) / 2.0 / a0, -(1.0 + cos_w) / a0, (1.0 + cos_w) / 2.0 / a0,
                   (-2.0 * cos_w) / a0, (1.0 - alpha) / a0, cutoff, "high-pass")

    @classmethod
    def low_pass(cls, cutoff: float, rate: int) -> "Biquad":
        w, alpha, cutoff = cls._omega(cutoff, rate)
        cos_w = math.cos(w)
        a0 = 1.0 + alpha
        return cls((1.0 - cos_w) / 2.0 / a0, (1.0 - cos_w) / a0, (1.0 - cos_w) / 2.0 / a0,
                   (-2.0 * cos_w) / a0, (1.0 - alpha) / a0, cutoff, "low-pass")

    # ── run ──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._z1 = self._z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a float array, carrying state on from the previous call."""
        return np.asarray(self.process_list(x.tolist()), dtype=np.float64)

    def process_list(self, xs: list) -> list:
        """The recursion itself. A biquad is sequential, so this is a Python loop,
        but over plain floats: indexing a numpy array element by element costs
        several times more than iterating a list, and this runs in an audio
        callback where a millisecond matters."""
        z1, z2 = self._z1, self._z2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        out = []
        append = out.append
        for xi in xs:                           # transposed direct form II
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            append(yi)
        self._z1, self._z2 = z1, z2
        return out

    def process_bytes(self, pcm: bytes) -> bytes:
        """int16 bytes in, int16 bytes out, clipped rather than wrapped."""
        if not pcm:
            return pcm
        y = self.process_list(np.frombuffer(pcm, dtype=np.int16).tolist())
        return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


class MicFilter:
    """The microphone chain: an optional high-pass, an optional low-pass, or neither.

    Rebuilt when a cutoff or the sample rate changes, so the control page can move
    either one while the loop is running. A cutoff of 0 switches that stage off.
    """

    def __init__(self, rate: int, high_pass: float = 0.0, low_pass: float = 0.0):
        self.rate = int(rate)
        self._hp_hz = 0.0
        self._lp_hz = 0.0
        self.hp: Optional[Biquad] = None
        self.lp: Optional[Biquad] = None
        self.configure(high_pass, low_pass)

    @property
    def high_pass(self) -> float:
        return self._hp_hz

    @property
    def low_pass(self) -> float:
        return self._lp_hz

    @property
    def active(self) -> bool:
        return self.hp is not None or self.lp is not None

    def configure(self, high_pass: Optional[float] = None, low_pass: Optional[float] = None,
                  rate: Optional[int] = None) -> None:
        if rate is not None and int(rate) != self.rate:
            self.rate = int(rate)
            high_pass = self._hp_hz if high_pass is None else high_pass
            low_pass = self._lp_hz if low_pass is None else low_pass
        if high_pass is not None and float(high_pass) != self._hp_hz:
            self._hp_hz = max(0.0, float(high_pass))
            self.hp = Biquad.high_pass(self._hp_hz, self.rate) if self._hp_hz > 0 else None
        if low_pass is not None and float(low_pass) != self._lp_hz:
            self._lp_hz = max(0.0, float(low_pass))
            self.lp = Biquad.low_pass(self._lp_hz, self.rate) if self._lp_hz > 0 else None

    def reset(self) -> None:
        for f in (self.hp, self.lp):
            if f is not None:
                f.reset()

    def process_bytes(self, pcm: bytes) -> bytes:
        if self.hp is not None:
            pcm = self.hp.process_bytes(pcm)
        if self.lp is not None:
            pcm = self.lp.process_bytes(pcm)
        return pcm

    def describe(self) -> str:
        parts = []
        if self.hp is not None:
            parts.append(f"high-pass {self.hp.cutoff:.0f} Hz")
        if self.lp is not None:
            parts.append(f"low-pass {self.lp.cutoff:.0f} Hz")
        return ", ".join(parts) if parts else "off"
