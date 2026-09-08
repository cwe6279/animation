"""
body.py — the character's body: movements asked for with {{move <name>}}.

Only a desktop stand-in for now. It knows the face's list of moves (face.json
"body": {"moves": [...]}) and logs each one; a real driver (servos on a PCA9685
over I2C on the Pi, or a serial link to an Arduino) implements the same
move(name, params) and is chosen here. See ROADMAP.md, Body.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from .actions import Action


class NullBody:
    def __init__(self, moves: Iterable[str] = ()):
        self.moves: List[str] = [str(m).lower() for m in moves]
        self.last: Optional[str] = None

    def move(self, name: str, params: dict) -> None:
        if self.moves and name not in self.moves:
            print(f"[body] unknown move '{name}' (have: {', '.join(self.moves)})")
            return
        self.last = name
        print(f"[body] move {name}{' ' + str(params) if params else ''}")

    def handler(self) -> Callable[[Action], None]:
        def run(a: Action) -> None:
            self.move(a.name, a.params)
        return run
