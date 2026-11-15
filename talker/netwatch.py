"""
netwatch.py — notice when the internet goes and comes back, and try to heal it.

Every cloud stage (speech-to-text, the brain, the voice, vision, the agent) dies
quietly when the network does: the character goes deaf and nobody is told. One
daemon thread opens a TCP connection to the services it depends on every
`interval` seconds. After `fails_to_offline` misses in a row it calls on_offline()
once; when a check succeeds again it calls on_online(seconds_down). While offline
it runs `heal_cmd` (by default `nmcli device connect wlan0`, which re-activates the
best saved network) every `heal_after` seconds and logs what came back. (It used to
be `nmcli device reconnect`, which nmcli does not have: every heal failed with
"argument 'reconnect' not understood", including through a 31-minute outage.)

    net = NetWatch(on_offline=lambda: ..., on_online=lambda s: ...)
    net.start()
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from typing import Callable, Optional, Sequence, Tuple

DEFAULT_HOSTS: Tuple[Tuple[str, int], ...] = (("api.anthropic.com", 443), ("api.elevenlabs.io", 443))


class NetWatch:
    def __init__(self, hosts: Sequence[Tuple[str, int]] = DEFAULT_HOSTS, interval: float = 10.0,
                 fails_to_offline: int = 2, on_offline: Optional[Callable[[], None]] = None,
                 on_online: Optional[Callable[[float], None]] = None, heal_after: float = 120.0,
                 heal_cmd: Optional[Sequence[str]] = ("nmcli", "--wait", "25", "device", "connect", "wlan0"),
                 probe: Optional[Callable[[], bool]] = None, clock: Callable[[], float] = time.monotonic):
        self.hosts = list(hosts)
        self.interval = interval
        self.fails_to_offline = fails_to_offline
        self.on_offline = on_offline or (lambda: None)
        self.on_online = on_online or (lambda s: None)
        self.heal_after = heal_after
        self.heal_cmd = list(heal_cmd) if heal_cmd else None
        self.probe = probe or self._probe
        self.clock = clock
        self.online = True
        self.down_since: Optional[float] = None
        self.outages = 0
        self._fails = 0
        self._last_heal = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _probe(self) -> bool:
        for host, port in self.hosts:
            try:
                socket.create_connection((host, port), timeout=4).close()
                return True
            except OSError:
                continue
        return False

    def check(self) -> None:
        """One probe and its consequences; public so tests can drive it without the thread."""
        if self.probe():
            self._fails = 0
            if not self.online:
                down = self.clock() - (self.down_since or self.clock())
                self.online, self.down_since = True, None
                print(f"[net] back online after {down:.0f}s")
                self.on_online(down)
            return
        self._fails += 1
        now = self.clock()
        if self.online and self._fails >= self.fails_to_offline:
            self.online, self.down_since, self._last_heal = False, now, now
            self.outages += 1
            print("[net] offline: the cloud services are unreachable")
            self.on_offline()
        elif not self.online and self.heal_cmd and now - self._last_heal >= self.heal_after:
            self._last_heal = now
            try:
                r = subprocess.run(self.heal_cmd, capture_output=True, text=True, timeout=30)
                print(f"[net] tried to heal ({' '.join(self.heal_cmd)}): "
                      f"{(r.stdout + r.stderr).strip()[:160] or 'exit ' + str(r.returncode)}")
            except Exception as e:
                print(f"[net] heal command failed: {e}")

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="netwatch", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except Exception as e:                     # never let the watchdog die
                print(f"[net] check failed: {e}")
