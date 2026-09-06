"""Live edge-tts test: needs internet + ffmpeg. Skipped automatically otherwise."""
import sys, os, socket, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from tts_backends import find_ffmpeg


def _online():
    try:
        socket.create_connection(("speech.platform.bing.com", 443), timeout=3).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not (_online() and find_ffmpeg()), reason="needs internet + ffmpeg")


def test_edge_backend_streams_audio_and_words():
    from audio_engine import NullAudioEngine
    from phoneme_scheduler import ScheduleReader
    from speech_pipeline import SpeechPipeline
    from tts_backends import EdgeTTSBackend
    eng = NullAudioEngine()
    sched = ScheduleReader()
    p = SpeechPipeline(eng, sched, EdgeTTSBackend())
    p.start()
    try:
        t0 = time.time()
        p.speak("[happy]Hello there, this is a live test. [sad]Second sentence follows here.")
        while p._active_sessions and time.time() - t0 < 20:
            time.sleep(0.05)
        assert p.stats["words"] >= 10
        assert p.stats["time_to_first_audio_ms"] < 2000
        assert sched.end_time > 2.0
        assert p.stats["audio_seconds"] > 2.0
        print("\nedge stats:", p.stats)
    finally:
        p.stop()
