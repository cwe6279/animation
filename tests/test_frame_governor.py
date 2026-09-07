import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frame_governor import FrameGovernor


def test_steps_down_under_load_and_recovers():
    changes = []
    g = FrameGovernor(60, window=10, recover_after_s=1.0, on_change=lambda o, n, a: changes.append((o, n)))
    for _ in range(10):
        g.record(0.020, 1 / 60)          # 20 ms of work cannot make 60 fps
    assert g.fps == 45 and changes[-1] == (60, 45)
    for _ in range(10):
        g.record(0.030, 1 / 45)          # still too slow for 45
    assert g.fps == 30
    for _ in range(10):
        g.record(0.030, 1 / 30)          # 30 ms fits 30 fps (33 ms budget)
    assert g.fps == 30
    for _ in range(200):
        g.record(0.005, 1 / 30)          # plenty of headroom for a while -> back up
    assert g.fps == 60 and changes[-1][1] == 60


def test_never_below_min_and_disabled_is_inert():
    g = FrameGovernor(60, min_fps=30, window=5, on_change=lambda *a: None)
    for _ in range(40):
        g.record(0.5, 1 / 30)
    assert g.fps == 30
    g2 = FrameGovernor(60, enabled=False, window=5)
    for _ in range(40):
        g2.record(0.5, 1 / 60)
    assert g2.fps == 60
