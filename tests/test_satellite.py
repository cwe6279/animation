"""A fake xiaozhi device talks to the satellite server: config, hello, audio both ways."""
import asyncio
import json
import socket
import time

import numpy as np
import pytest

pytest.importorskip("opuslib")
aiohttp = pytest.importorskip("aiohttp")

from talker.satellite import (SatelliteServer, SatelliteAudioEngine, SpeakerShaper, DEVICE_MIC_RATE,
                              DEVICE_OUT_RATE, FRAME_MS, OTA_PATH, WS_PATH)
import opuslib


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    port = _free_port()
    events = []
    srv = SatelliteServer(host="127.0.0.1", port=port, on_event=lambda k, t: events.append((k, t)))
    eng = SatelliteAudioEngine(srv, lookahead=5.0)      # send everything at once in tests
    srv.start()
    for _ in range(100):                                # wait for the port
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    return srv, eng, events, port


async def _device(port, eng, received):
    """One device session: OTA check-in, hello, a 60 ms mic frame, then listen for speech."""
    base = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        async with s.post(base + OTA_PATH, json={"application": {"version": "9.9.9"},
                                                 "board": {"type": "fake", "ip": "10.0.0.2"}},
                          headers={"Device-Id": "aa:bb", "Client-Id": "c1"}) as r:
            cfg = await r.json()
        assert cfg["websocket"]["url"].endswith(WS_PATH)
        assert cfg["firmware"]["version"] == "9.9.9"           # "current is the latest"
        assert "server_time" in cfg and "activation" not in cfg

        async with s.ws_connect(cfg["websocket"]["url"]) as ws:
            await ws.send_json({"type": "hello", "version": 1, "transport": "websocket",
                                "audio_params": {"format": "opus", "sample_rate": DEVICE_MIC_RATE,
                                                 "channels": 1, "frame_duration": FRAME_MS}})
            hello = json.loads((await ws.receive()).data)
            assert hello["type"] == "hello" and hello["audio_params"]["sample_rate"] == DEVICE_OUT_RATE

            # the device starts listening and sends one frame of a 440 Hz tone
            await ws.send_json({"type": "listen", "state": "start", "mode": "auto"})
            await asyncio.sleep(0.5)                       # the first 0.4 s after "listen start" are muted
            enc = opuslib.Encoder(DEVICE_MIC_RATE, 1, opuslib.APPLICATION_VOIP)
            n = DEVICE_MIC_RATE * FRAME_MS // 1000
            tone = (np.sin(np.arange(n) * 2 * np.pi * 440 / DEVICE_MIC_RATE) * 8000).astype(np.int16)
            await ws.send_bytes(enc.encode(tone.tobytes(), n))

            # the host speaks half a second
            await asyncio.sleep(0.1)
            frames = DEVICE_OUT_RATE // 2
            speech = (np.sin(np.arange(frames) * 2 * np.pi * 440 / DEVICE_OUT_RATE) * 8000).astype(np.int16)
            eng.enqueue_pcm(speech.tobytes(), DEVICE_OUT_RATE)
            eng.note_sentence("Hello there.", 0.0)

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                msg = await ws.receive(timeout=1.0)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    received.append(("audio", msg.data))
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    received.append(("json", d))
                    payload = d.get("payload") or {}
                    if d.get("type") == "mcp" and payload.get("method") == "tools/call" \
                            and payload["params"]["name"] == "self.get_device_status":
                        status = {"battery": {"level": 73, "charging": True}, "network": {"signal": "strong"},
                                  "chip": {"temperature": 41.5}}
                        await ws.send_json({"type": "mcp", "payload": {"jsonrpc": "2.0", "id": payload["id"], "result": {
                            "content": [{"type": "text", "text": json.dumps(status)}], "isError": False}}})
                    if d.get("type") == "tts" and d.get("state") == "stop":
                        break
                else:
                    break


def test_device_round_trip(server):
    srv, eng, events, port = server
    mic = []
    eng.start_mic(on_frames=mic.append, rate=16000)
    received = []
    asyncio.run(_device(port, eng, received))

    # the device's mic frame reached the loop as 60 ms of 16 kHz int16
    assert len(mic) == 1 and len(mic[0]) == DEVICE_MIC_RATE * FRAME_MS // 1000 * 2

    kinds = [(k, d.get("type"), d.get("state")) for k, d in received if k == "json"]
    assert ("json", "tts", "start") in kinds
    assert ("json", "tts", "sentence_start") in kinds
    assert ("json", "tts", "stop") in kinds
    assert kinds.index(("json", "tts", "start")) < kinds.index(("json", "tts", "stop"))
    audio = [d for k, d in received if k == "audio"]
    assert 8 <= len(audio) <= 10                      # 0.5 s in 60 ms frames (8.33 -> 9 padded)
    dec = opuslib.Decoder(DEVICE_OUT_RATE, 1)
    pcm = np.frombuffer(dec.decode(audio[1], DEVICE_OUT_RATE * FRAME_MS // 1000), dtype=np.int16)
    assert np.abs(pcm).max() > 2000                    # a tone, not silence
    subtitle = [d for k, d in received if k == "json" and d.get("state") == "sentence_start"][0]
    assert subtitle["text"] == "Hello there."
    assert any("checked in" in t for k, t in events)
    assert srv.device_status["battery"]["level"] == 73
    assert srv.status_line() == "the board reports battery 73%, charging, wifi signal strong, chip 42 C"
    srv.device_status = {"battery": {"level": 41, "charging": False, "voltage": 3.812}}
    assert srv.status_line() == "the board reports battery 41% (3.81 V), on battery"


def test_speech_dropped_without_device(server):
    srv, eng, events, port = server
    t = eng.enqueue_pcm(b"\0\0" * 2400, DEVICE_OUT_RATE)
    assert t >= 0 and eng.queued_seconds() > 0.05
    assert not srv.connected


async def _paced_device(port, eng, arrivals):
    """A device that records when each audio frame arrives, for a 3 s sentence sent at once."""
    base = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(base + WS_PATH) as ws:
            await ws.send_json({"type": "hello", "version": 1, "transport": "websocket",
                                "audio_params": {"format": "opus", "sample_rate": DEVICE_MIC_RATE,
                                                 "channels": 1, "frame_duration": FRAME_MS}})
            await ws.receive()
            await ws.send_json({"type": "listen", "state": "start", "mode": "auto"})
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            frames = DEVICE_OUT_RATE * 3
            tone = (np.sin(np.arange(frames) * 2 * np.pi * 440 / DEVICE_OUT_RATE) * 8000).astype(np.int16)
            eng.enqueue_pcm(tone.tobytes(), DEVICE_OUT_RATE)        # one chunk: a whole sentence
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                msg = await ws.receive(timeout=2.0)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    arrivals.append(time.monotonic() - t0)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    if json.loads(msg.data).get("state") == "stop":
                        break
                else:
                    break


def test_frames_are_paced_not_burst():
    """The device buffers ~1.2 s and drops the rest, so a sentence must leave at playback pace."""
    port = _free_port()
    srv = SatelliteServer(host="127.0.0.1", port=port, on_event=lambda k, t: None)
    eng = SatelliteAudioEngine(srv, lookahead=0.5)
    srv.start()
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    arrivals = []
    asyncio.run(_paced_device(port, eng, arrivals))
    assert 50 <= len(arrivals) <= 51                       # 3 s in 60 ms frames
    # no frame leaves more than lookahead + start hold ahead of its moment on the timeline
    lead = max(i * FRAME_MS / 1000 - t for i, t in enumerate(arrivals))
    assert lead <= 0.75, f"a frame left {lead:.2f} s early: a burst"
    assert arrivals[-1] >= 2.3                             # the last frame left about when it was due


def _tone(hz, seconds=1.0, amp=8000, rate=DEVICE_OUT_RATE):
    return (np.sin(np.arange(int(rate * seconds)) * 2 * np.pi * hz / rate) * amp).astype(np.int16)


def test_speaker_shaper_cuts_bass_lifts_speech_and_limits():
    """Bass the small speaker cannot play is removed; the voice band is lifted; peaks stay under the ceiling."""
    rate = DEVICE_OUT_RATE
    sh = SpeakerShaper(rate, hpf_hz=250.0, boost_db=6.0)
    bass = np.frombuffer(sh.process(_tone(80).tobytes()), dtype=np.int16)[rate // 2:]
    assert np.abs(bass).max() < 8000 * 0.1                 # more than 20 dB down
    sh = SpeakerShaper(rate, hpf_hz=250.0, boost_db=6.0)
    voice = np.frombuffer(sh.process(_tone(1000, amp=4000).tobytes()), dtype=np.int16)[rate // 2:]
    assert 7000 < np.abs(voice).max() <= 8200             # +6 dB, no limiting needed
    sh = SpeakerShaper(rate, hpf_hz=250.0, boost_db=6.0)
    loud = np.frombuffer(sh.process(_tone(1000, amp=30000).tobytes()), dtype=np.int16)
    assert np.abs(loud).max() <= 0.9 * 32767 + 1          # limited, not clipped
    # chunk boundaries are seamless: two halves equal the whole
    whole = SpeakerShaper(rate, boost_db=0.0).process(_tone(500).tobytes())
    sh = SpeakerShaper(rate, boost_db=0.0)
    t = _tone(500).tobytes()
    halves = sh.process(t[:len(t) // 2 + 2]) + sh.process(t[len(t) // 2 + 2:])
    assert halves == whole
    assert not SpeakerShaper(rate, hpf_hz=0.0, boost_db=0.0).enabled


def test_board_event_reaches_the_loop_and_arms_the_idle_close():
    import asyncio
    from talker import satellite as sat
    server = sat.SatelliteServer.__new__(sat.SatelliteServer)
    events = []
    server.on_event = lambda kind, text: events.append((kind, text))
    got = []
    server.on_board_event = lambda kind, text: got.append((kind, text))

    class Eng:
        _idle_deadline = None
        idle_close_s = 40.0
        _clock = staticmethod(lambda: 100.0)
        device_event = sat.SatelliteAudioEngine.device_event
    server.engine = Eng()
    asyncio.run(server._on_json(None, {"type": "event", "kind": "picked_up", "text": "picked up"}))
    assert got == [("picked_up", "picked up")]
    assert ("satellite", "board event: picked up") in events
    assert server.engine._idle_deadline == 140.0


def _bare_engine(now):
    from talker import satellite as sat
    server = sat.SatelliteServer.__new__(sat.SatelliteServer)
    server.on_event = lambda kind, text: None
    return sat.SatelliteAudioEngine(server, clock=lambda: now[0])


def test_end_conversation_while_quiet_closes_at_once():
    """'Stop listening' after she has finished: the board goes back to its wake word now,
    not after the 30 s idle timeout."""
    now = [100.0]
    eng = _bare_engine(now)
    eng.end_conversation()
    now[0] += 0.01
    assert ("close", None) in eng.tick()


def test_end_conversation_waits_for_her_last_words():
    now = [100.0]
    eng = _bare_engine(now)
    eng.speaking = True                       # her farewell is still playing
    eng.end_conversation()
    eng.busy = lambda: True
    now[0] += 5.0
    assert ("close", None) not in eng.tick()   # never cut her off mid-word
    eng.busy = lambda: False
    out = eng.tick()                           # the farewell has played: tts stop ...
    assert ("json", {"type": "tts", "state": "stop"}) in out
    now[0] += 0.01
    assert ("close", None) in eng.tick()       # ... and the channel closes right after


def test_without_end_the_usual_idle_timeout_applies():
    now = [100.0]
    eng = _bare_engine(now)
    eng.speaking = True
    out = eng.tick()
    assert ("json", {"type": "tts", "state": "stop"}) in out
    now[0] += 1.0
    assert ("close", None) not in eng.tick()
    now[0] += eng.idle_close_s
    assert ("close", None) in eng.tick()
