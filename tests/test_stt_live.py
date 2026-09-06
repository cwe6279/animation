"""Vosk on real speech: synthesizes a sentence with edge-tts, decodes to 16 kHz, transcribes.
Skipped without internet, ffmpeg, vosk, or the model."""
import sys, os, socket, subprocess, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


def _online():
    try:
        socket.create_connection(("speech.platform.bing.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


def _vosk_ready():
    try:
        import vosk  # noqa
        from stt_backends import VOSK_MODEL_NAME, CACHE_DIR
        return os.path.isdir(os.path.join(CACHE_DIR, VOSK_MODEL_NAME))
    except ImportError:
        return False


from tts_backends import find_ffmpeg
pytestmark = pytest.mark.skipif(not (_online() and find_ffmpeg() and _vosk_ready()),
                                reason="needs internet + ffmpeg + vosk model")


def synth_pcm16k(text: str) -> bytes:
    import edge_tts
    async def run():
        mp3 = b""
        async for ch in edge_tts.Communicate(text, voice="en-US-GuyNeural").stream():
            if ch["type"] == "audio":
                mp3 += ch["data"]
        return mp3
    mp3 = asyncio.run(run())
    return subprocess.run([find_ffmpeg(), "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                           "-ar", "16000", "-ac", "1", "pipe:1"], input=mp3, capture_output=True).stdout


def test_vosk_transcribes_synth_speech():
    from stt_backends import VoskSTT
    from voice_loop import VoiceLoop
    stt = VoskSTT()
    pcm = synth_pcm16k("What is the weather like on Mars today?") + bytes(16000 * 2)  # + 1 s silence
    heard = []
    class Spk:
        spoken = []
        is_busy = False
        def speak_stream(self, chunks): self.spoken.append("".join(chunks))
        def interrupt(self): pass
    loop = VoiceLoop(stt, lambda t: iter(["ok"]), Spk(), on_event=lambda k, s: heard.append((k, s)))
    for i in range(0, len(pcm), 3200):        # 100 ms chunks, like a mic
        loop._process(pcm[i:i + 3200])
    finals = [s for k, s in heard if k == "you"]
    assert finals, heard
    text = finals[0].lower()
    print("\nvosk heard:", text)
    assert "weather" in text and "mars" in text
