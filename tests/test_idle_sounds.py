"""Idle sounds: only when quiet long enough, at random gaps, never the same twice in a row."""
import random

from talker.idle_sounds import IdleSounds


class Bank:
    def __init__(self, names):
        self.names = sorted(names)
        self.played = []

    def play(self, name):
        self.played.append(name)
        return True


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


def test_idle_sounds_disabled_without_files():
    idle = IdleSounds(Bank([]), lambda: True, clock=lambda: 1e9)
    assert not idle.enabled and idle.tick() is None
