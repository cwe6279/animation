"""Cloud hearing that fails while the internet works: an out-of-credit account."""
import asyncio
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

aiohttp = pytest.importorskip("aiohttp")

from talker import stt_backends


class FakeMsg:
    def __init__(self, type_, data="", extra=""):
        self.type, self.data, self.extra = type_, data, extra


class FakeWS:
    """What ElevenLabs does with no credit: session_started, quota_exceeded, close."""
    close_code = 1000

    def __init__(self, script):
        self.script = list(script)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.script:
            raise StopAsyncIteration
        return self.script.pop(0)

    async def send_json(self, d):
        pass


def fake_session(scripts, connects):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def ws_connect(self, url, headers=None, heartbeat=None):
            connects.append(time.monotonic())
            return FakeWS(scripts.pop(0) if scripts else NO_CREDIT)
    return Session


T = aiohttp.WSMsgType
NO_CREDIT = [FakeMsg(T.TEXT, '{"message_type": "session_started"}'),
             FakeMsg(T.TEXT, '{"message_type": "quota_exceeded", "error": "You have exceeded your quota."}'),
             FakeMsg(T.CLOSE, extra="insufficient_funds_initial_check")]
WORKS = [FakeMsg(T.TEXT, '{"message_type": "session_started"}'),
         FakeMsg(T.TEXT, '{"message_type": "committed_transcript", "text": "hello Clara"}'),
         FakeMsg(T.CLOSE)]


def test_out_of_credit_is_reported_once_backs_off_and_recovers(monkeypatch):
    connects, problems, recovered = [], [], []
    scripts = [NO_CREDIT, NO_CREDIT, WORKS]
    monkeypatch.setattr(aiohttp, "ClientSession", fake_session(scripts, connects))
    waits = []

    real_sleep = asyncio.sleep
    test_loop = {}

    async def fast_sleep(s):
        waits.append(s)
        await real_sleep(0)

    def patched_sleep(s, *a, **k):
        # Only this test's own loop gets the instant sleep. Other tests leave servers running
        # in threads with their own loops (the satellite's 20 ms ticker); patching sleep for
        # them too turned the ticker into a hot loop that appended to `waits` without end,
        # 4 MB/s, and a stuck run of this test grew to 29 GB.
        try:
            mine = asyncio.get_running_loop() is test_loop.get("loop")
        except RuntimeError:
            mine = False
        if not mine or s == 0:
            return real_sleep(s, *a, **k)
        return fast_sleep(s)
    monkeypatch.setattr(stt_backends.ElevenLabsSTT, "QUOTA_RETRY_S", 300.0)
    monkeypatch.setattr(asyncio, "sleep", patched_sleep)

    stt = stt_backends.ElevenLabsSTT.__new__(stt_backends.ElevenLabsSTT)
    stt.sample_rate, stt.silence_s, stt.language, stt.api_key = 16000, 0.9, "en", "k"
    stt.speech_active, stt.problem = False, ""
    stt.on_problem = lambda kind, detail: problems.append((kind, detail))
    stt.on_recovered = recovered.append
    import queue
    stt._results, stt._audio = queue.Queue(), asyncio.Queue()

    stt.gc_ok = lambda: True

    async def run():
        test_loop["loop"] = asyncio.get_running_loop()
        stt._awake, stt._ws, stt._want = asyncio.Event(), None, asyncio.Event()
        stt._awake.set()
        task = asyncio.ensure_future(stt._session_forever())

        async def someone_talking():
            # hearing connects only when there is audio to send (the Pi change): keep some coming
            while True:
                if stt._audio.empty():
                    stt._audio.put_nowait(b"\0" * 3200)
                    stt._want.set()
                await real_sleep(0)
        voice = asyncio.ensure_future(someone_talking())

        async def until_recovered():
            while not recovered:
                await real_sleep(0)
        try:
            await asyncio.wait_for(until_recovered(), timeout=10)   # never hang the suite again
        finally:
            task.cancel()
            voice.cancel()
    asyncio.run(run())

    assert problems == [("out_of_credit", "quota_exceeded: You have exceeded your quota.")]   # once
    assert waits[0] >= 300 and waits[1] >= 300          # not every second: every five minutes
    assert recovered == ["out_of_credit"] and stt.problem == ""
    assert stt._results.get_nowait().text == "hello Clara"


def test_mic_audio_never_piles_up_while_hearing_is_down():
    # 32 KB/s queued with nothing draining it filled the Pi's memory and swap in 3 hours.
    import queue
    stt = stt_backends.ElevenLabsSTT.__new__(stt_backends.ElevenLabsSTT)
    stt._results, stt._ws = queue.Queue(), None
    stt._loop = asyncio.new_event_loop()
    stt._audio, stt._want = asyncio.Queue(), asyncio.Event()
    frame = b"\0" * 3200
    for _ in range(500):
        stt.feed(frame)          # disconnected: kept to connect on demand, but never past the cap
    stt._loop.run_until_complete(asyncio.sleep(0))
    assert stt._audio.qsize() == stt.MAX_QUEUED and stt._want.is_set()
    stt._ws = object()                                   # connected, but the sender is stuck
    for _ in range(500):
        stt.feed(frame)
    stt._loop.run_until_complete(asyncio.sleep(0))
    assert stt._audio.qsize() == stt.MAX_QUEUED
