"""The control page: state, live tunables, actions, reference, all over plain HTTP."""
import argparse
import json
import urllib.request

from talker.web_panel import WebPanel, describe_parser


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_panel_serves_state_tunables_and_actions():
    box = {"silence": 600, "spoken": []}
    panel = WebPanel(port=0, host="127.0.0.1")
    panel.status_fn = lambda: {"face": "pumpkin", "mic_rms": 42}
    panel.tunable("silence_ms", lambda: box["silence"], lambda v: box.__setitem__("silence", v),
                  "pause that ends a turn", kind="int", unit="ms", lo=150, hi=2000, flag="--silence-ms")
    panel.action("speak", lambda t: (box["spoken"].append(t), "speaking")[1], "say it", takes_text=True)
    p = argparse.ArgumentParser()
    p.add_argument("--silence-ms", type=int, default=600, help="pause (default %(default)s)")
    panel.parser, panel.args = p, p.parse_args(["--silence-ms", "450"])
    url = panel.start()
    assert url and panel.port > 0
    base = f"http://127.0.0.1:{panel.port}"
    try:
        code, ctype, body = _get(base + "/")
        assert code == 200 and "text/html" in ctype and b"Talker control" in body
        st = json.loads(_get(base + "/api/state")[2])
        assert st["status"]["face"] == "pumpkin" and st["tunables"][0]["value"] == 600
        assert st["tunables"][0]["flag"] == "--silence-ms" and "pause" in st["tunables"][0]["help"]
        r = _post(base + "/api/set", {"name": "silence_ms", "value": "100"})     # clamped to lo
        assert r["value"] == 150 and box["silence"] == 150
        r = _post(base + "/api/action", {"name": "speak", "text": "hello"})
        assert r["message"] == "speaking" and box["spoken"] == ["hello"]
        ref = json.loads(_get(base + "/api/reference")[2])
        assert ref["flags"][0]["flags"] == "--silence-ms" and ref["flags"][0]["value"] == "450"
        assert "default 600" in ref["flags"][0]["help"]
        assert _post(base + "/api/set", {"name": "nope", "value": 1})["error"]
        assert any(e["kind"] == "panel" for e in json.loads(_get(base + "/api/state")[2])["events"])
    finally:
        panel.stop()


def test_describe_parser_handles_percent_and_bools():
    p = argparse.ArgumentParser()
    p.add_argument("--vision-change", type=float, default=0.035, help="3.5%% change (default %(default)s)")
    p.add_argument("--barge-in", action="store_true", help="interrupt")
    rows = describe_parser(p, p.parse_args([]))
    assert rows[0]["help"] == "3.5% change (default 0.035)" and rows[1]["value"] == "False"


def test_nmcli_fields_unescape_colons():
    from talker.web_panel import WifiControl
    assert WifiControl._fields("no:HomeNet:100:WPA2 WPA3") == ["no", "HomeNet", "100", "WPA2 WPA3"]
    assert WifiControl._fields(r"yes:Cafe\:Wifi:80:WPA2") == ["yes", "Cafe:Wifi", "80", "WPA2"]
