import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talker.calibrate import recommend


def test_recommend_good_separation():
    r = recommend(ambient=300, speaker=800, person=5000)
    assert r["barge_in_ok"] and r["mic_gain"] == "ok"
    thr = r["onset_threshold"]
    assert 800 < r["barge_in_boost"] * thr < 5000            # threshold lands between bleed and person


def test_recommend_marginal_and_bad():
    assert not recommend(300, 3000, 4500)["barge_in_ok"]
    assert "not viable" in recommend(300, 4000, 4200)["verdict"]
    assert recommend(300, 500, 300)["mic_gain"] == "raise"
    assert recommend(300, 500, 30000)["mic_gain"] == "lower"


def test_boost_is_clamped():
    assert 1.0 <= recommend(50, 100, 200)["barge_in_boost"] <= 8.0
    assert recommend(5000, 100000, 200000)["barge_in_boost"] <= 8.0
