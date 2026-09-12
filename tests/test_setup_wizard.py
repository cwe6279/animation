"""The setup wizard's judgement calls, and the checks that make a fresh clone work."""
import json

import pytest

from talker.setup_wizard import OK, BAD, SKIP, WARN, Report, device_name, save_devices


def test_report_collects_and_flags_only_real_failures():
    r = Report()
    r.add(OK, "numpy"); r.add(WARN, "ffmpeg missing"); r.add(SKIP, "camera")
    assert r.failed == []
    r.add(BAD, "no microphone", "nothing to listen with")
    assert [f[1] for f in r.failed] == ["no microphone"]


class FakeAudio:
    def list_input_devices(self):
        return [(7, "Samson Go Mic: USB Audio (hw:4,0)", 44100, False)]

    def list_output_devices(self):
        return [(16, "Jabra SPEAK 510 Analog Stereo", 48000, True)]


def test_a_chosen_index_becomes_a_name():
    """Indices move when anything is plugged in, so the command we print uses names."""
    a = FakeAudio()
    assert device_name(a, "7", "input") == "Samson Go Mic: USB Audio (hw:4,0)"
    assert device_name(a, "16", "output") == "Jabra SPEAK 510 Analog Stereo"
    assert device_name(a, "samson", "input") == "samson"      # already a fragment
    assert device_name(a, None, "input") is None
    assert device_name(a, "99", "input") == "99"              # unknown index, left alone


def test_saving_devices_keeps_the_measurements(tmp_path, monkeypatch):
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"ambient": 1052, "barge_in_boost": 1.7, "mic_device": "old"}))
    monkeypatch.setattr("talker.calibrate.CALIBRATION_FILE", str(cal))
    assert save_devices("samson go", "jabra")
    got = json.loads(cal.read_text())
    assert got["mic_device"] == "samson go" and got["output_device"] == "jabra"
    assert got["ambient"] == 1052 and got["barge_in_boost"] == 1.7   # not clobbered
    assert got["time"]


def test_ollama_rejects_a_model_that_is_not_pulled(monkeypatch):
    """A wrong --model used to surface a minute later as 'I could not think of an answer'."""
    import talker.brains.ollama_chat as oc
    monkeypatch.setattr(oc, "list_models", lambda host=None: ["qwen3:8b", "gemma4:31b"])
    with pytest.raises(RuntimeError, match="no model 'nope:404'"):
        oc.OllamaChat(model="nope:404", character="x")
    assert oc.OllamaChat(model="qwen3:8b", character="x").model == "qwen3:8b"

    def down(host=None):
        raise OSError("connection refused")
    monkeypatch.setattr(oc, "list_models", down)
    assert oc.OllamaChat(model="anything", character="x").model == "anything"


def test_faster_whisper_is_a_hard_requirement():
    """It is the default recognizer; commenting it out broke every fresh clone."""
    reqs = open("requirements.txt").read()
    assert "\nfaster-whisper>=1.0" in reqs, "faster-whisper must not be commented out"
