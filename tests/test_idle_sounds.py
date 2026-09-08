"""Idle sounds: only when quiet long enough, at random gaps, never the same twice in a row."""
import random

from talker.idle_sounds import IdleSounds


class Channel:
    def __init__(self): self.busy, self.faded = True, None
    def get_busy(self): return self.busy
    def fadeout(self, ms): self.faded, self.busy = ms, False


class Bank:
    def __init__(self, names):
        self.names = sorted(names)
        self.played = []
        self.channels = []

    def play(self, name):
        self.played.append(name)
        ch = Channel(); self.channels.append(ch)
        return ch


def test_idle_sounds_wait_for_quiet_and_space_out():
    t = {"now": 0.0}
    quiet = {"v": False}
    bank = Bank(["purr", "meow"])
    idle = IdleSounds(bank, lambda: quiet["v"], interval=(20, 20), quiet_for=10,
                      clock=lambda: t["now"], rng=random.Random(1))
    for _ in range(30):                       # noisy room: nothing
        t["now"] += 1; idle.tick()
    assert bank.played == []
    quiet["v"] = True
    for _ in range(9):                        # quiet, but not for long enough yet
        t["now"] += 1; idle.tick()
    assert bank.played == []
    t["now"] += 2; idle.tick()
    assert len(bank.played) == 1
    for _ in range(19):                       # the next one waits the full gap
        t["now"] += 1; idle.tick()
    assert len(bank.played) == 1
    t["now"] += 1; idle.tick()
    assert len(bank.played) == 2 and bank.played[0] != bank.played[1]
    quiet["v"] = False; t["now"] += 25; idle.tick()      # someone talks: silence again
    assert len(bank.played) == 2
    assert bank.channels[-1].faded == 250                # and the long purr fades out at once
    quiet["v"] = True; idle.hush()                       # a reply starting hushes it too


def test_idle_sounds_disabled_without_files():
    idle = IdleSounds(Bank([]), lambda: True, clock=lambda: 1e9)
    assert not idle.enabled and idle.tick() is None
