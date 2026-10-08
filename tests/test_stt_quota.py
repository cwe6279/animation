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

    async def fast_sleep(s):
        waits.append(s)
        await asyncio.sleep(0)
    monkeypatch.setattr(stt_backends.ElevenLabsSTT, "QUOTA_RETRY_S", 300.0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0) if s == 0 else fast_sleep(s))

    stt = stt_backends.ElevenLabsSTT.__new__(stt_backends.ElevenLabsSTT)
    stt.sample_rate, stt.silence_s, stt.language, stt.api_key = 16000, 0.9, "en", "k"
    stt.speech_active, stt.problem = False, ""
    stt.on_problem = lambda kind, detail: problems.append((kind, detail))
    stt.on_recovered = recovered.append
    import queue
    stt._results, stt._audio = queue.Queue(), asyncio.Queue()

    async def run():
        stt._awake, stt._ws = asyncio.Event(), None
        stt._awake.set()
        task = asyncio.ensure_future(stt._session_forever())
        while not recovered:
            await real_sleep(0)
        task.cancel()
    asyncio.run(run())

    assert problems == [("out_of_credit", "quota_exceeded: You have exceeded your quota.")]   # once
    assert waits[0] >= 300 and waits[1] >= 300          # not every second: every five minutes
    assert recovered == ["out_of_credit"] and stt.problem == ""
    assert stt._results.get_nowait().text == "hello Clara"
