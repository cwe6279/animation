"""
actions.py — action blocks in the brain's reply, next to the speech.

    [happy] Welcome in! {{move nod}} Mind the step. {{sfx creak}}

The words go to the voice exactly as before; a {{...}} block is never spoken.
It is removed in the same step that strips the [emotion] tags, so the text
reaches the TTS with no extra hop, and the action fires when the words before
it are spoken, off the same word timeline the eyes use. Kinds:

    {{move <name>}}           body: a named movement (servos later; logged now)
    {{sfx <name>}}            a sound file from the face folder's sounds/ dir
    {{tool <name> <args>}}    a registered Python callable; whatever it returns
                              goes back to the brain as context for the next turn

The brain is only told about the kinds this face actually has (see
action_rules), so a face with no sounds folder and no body never sees them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .phoneme_scheduler import parse_tags

# {{kind name args}}; args may hold one level of JSON braces: {{move turn {"deg": 30}}}
_BLOCK_RE = re.compile(r"\{\{\s*([A-Za-z_]\w*)\b[:\s]*((?:[^{}]|\{[^{}]*\})*?)\s*\}\}")


@dataclass
class Action:
    kind: str                     # move | sfx | tool | anything a handler is registered for
    name: str                     # first word after the kind
    args: str = ""                # the rest, verbatim
    raw: str = ""                 # the block as written, for logs

    @property
    def params(self) -> dict:
        """args as a dict: JSON if it looks like JSON, else {"text": args}."""
        a = self.args.strip()
        if a.startswith("{"):
            try:
                return json.loads(a)
            except ValueError:
                pass
        return {"text": a} if a else {}


def parse_actions(text: str) -> Tuple[str, List[Tuple[int, Action]]]:
    """
    Remove {{...}} blocks. Returns (text without blocks, [(word_index, action)])
    where word_index is the index of the spoken word the block precedes, counted
    the same way parse_tags counts (emotion tags are not words).
    """
    actions: List[Tuple[int, Action]] = []
    out: List[str] = []
    pos = 0
    words = 0
    for m in _BLOCK_RE.finditer(text):
        before = text[pos:m.start()]
        out.append(before)
        words += len(parse_tags(before)[0].split())
        body = m.group(2).strip()
        name, _, args = body.partition(" ")
        actions.append((words, Action(kind=m.group(1).lower(), name=name.strip().lower(),
                                      args=args.strip(), raw=m.group(0))))
        pos = m.end()
    out.append(text[pos:])
    if not actions:
        return text, actions
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip(), actions


def strip_actions(text: str) -> str:
    return parse_actions(text)[0]


Handler = Callable[[Action], Optional[str]]


class ActionDispatcher:
    """kind -> handler. A handler may return a string; on_result gets it (used to
    feed tool output back to the brain)."""

    def __init__(self, on_result: Optional[Callable[[Action, str], None]] = None):
        self._handlers: Dict[str, Handler] = {}
        self.on_result = on_result
        self.history: List[Action] = []

    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind.lower()] = handler

    @property
    def kinds(self) -> List[str]:
        return sorted(self._handlers)

    def dispatch(self, action: Action) -> None:
        self.history.append(action)
        h = self._handlers.get(action.kind)
        if h is None:
            print(f"[action] ignored (no handler for '{action.kind}'): {action.raw}")
            return
        try:
            result = h(action)
        except Exception as e:                       # never let an action break speech
            print(f"[action] {action.raw} failed: {e}")
            return
        if result and self.on_result is not None:
            self.on_result(action, str(result))


class SoundBank:
    """Sound effects from a folder: {{sfx creak}} plays creak.wav / .ogg / .mp3.
    Mixed by pygame.mixer on the default output, so it can play over speech."""

    EXTS = (".wav", ".ogg", ".mp3")

    def __init__(self, directory: Optional[str]):
        self.dir = directory
        self.files: Dict[str, str] = {}
        if directory and os.path.isdir(directory):
            for fn in sorted(os.listdir(directory)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() in self.EXTS:
                    self.files[stem.lower()] = os.path.join(directory, fn)
        self._ready: Optional[bool] = None
        self._cache: Dict[str, object] = {}

    @property
    def names(self) -> List[str]:
        return sorted(self.files)

    def _path(self, name: str) -> Optional[str]:
        n = name.lower().strip()
        if n in self.files:
            return self.files[n]
        for stem, path in self.files.items():         # "laugh" matches laugh2
            if stem.startswith(n) or n.startswith(stem):
                return path
        return None

    def play(self, name: str):
        """Start the sound; returns the mixer channel (truthy) or None."""
        path = self._path(name)
        if path is None:
            print(f"[sfx] no sound named '{name}' in {self.dir}")
            return None
        import pygame
        if self._ready is None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._ready = True
            except Exception as e:
                print(f"[sfx] mixer unavailable: {e}")
                self._ready = False
        if not self._ready:
            return None
        snd = self._cache.get(path)
        if snd is None:
            snd = pygame.mixer.Sound(path)
            self._cache[path] = snd
        channel = snd.play()
        print(f"[sfx] {os.path.basename(path)}")
        return channel

    def handler(self) -> Handler:
        return lambda a: (self.play(a.name), None)[1]


@dataclass
class ToolBox:
    """Named Python callables the brain may invoke with {{tool name args}}.
    Each entry: name -> (description, fn(args: str) -> str | None)."""

    tools: Dict[str, Tuple[str, Callable[[str], Optional[str]]]] = field(default_factory=dict)

    def add(self, name: str, description: str, fn: Callable[[str], Optional[str]]) -> None:
        self.tools[name.lower()] = (description, fn)

    @property
    def names(self) -> List[str]:
        return sorted(self.tools)

    def describe(self) -> str:
        return "; ".join(f"{n}: {d}" for n, (d, _) in sorted(self.tools.items()))

    def handler(self) -> Handler:
        def run(a: Action) -> Optional[str]:
            entry = self.tools.get(a.name)
            if entry is None:
                print(f"[tool] unknown tool '{a.name}'")
                return None
            print(f"[tool] {a.name} {a.args}".rstrip())
            return entry[1](a.args)
        return run


def action_rules(moves: List[str] = (), sounds: List[str] = (), tools: str = "") -> str:
    """The system-prompt addition for whatever this face can do; '' if nothing."""
    lines = []
    if moves:
        lines.append("Movements: {{move <name>}} with one of: " + ", ".join(moves) + ".")
    if sounds:
        lines.append("Sound effects: {{sfx <name>}} with one of: " + ", ".join(sounds) + ".")
    if tools:
        lines.append("Tools: {{tool <name> <arguments>}} where the tools are " + tools
                     + ". A tool's answer reaches you as an observation on the next turn, so "
                     "say you are checking rather than inventing the result.")
    if not lines:
        return ""
    return (
        "\n\nBesides speaking you can act. Write an action in double braces at the point in your "
        "words where it should happen. Actions are never spoken and fire as the words before them "
        "are said. Use at most one or two per reply, only when it adds to the moment, and never "
        "describe the action in words as well.\n" + "\n".join(lines)
        + "\nExample: [happy] Welcome in! {{move nod}} Mind the step."
    )
