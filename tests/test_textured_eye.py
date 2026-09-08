import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from talker.phoneme_scheduler import Emotion
from talker.textured_eye import EyeMotion, EyeMotionConfig, TexturedEye, TexturedEyeAssets

GOAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faces", "goat", "eye")


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def run(motion, clock, seconds, dt=1/60, emotion=Emotion.NEUTRAL):
    n = int(seconds / dt)
    for _ in range(n):
        clock.t += dt
        motion.update(dt, emotion)


def test_saccades_move_then_hold():
    clk = Clock(); m = EyeMotion(EyeMotionConfig(hold_s=(0.2, 0.2), move_s=(0.1, 0.1), micro=0.0), rng=np.random.default_rng(1), clock=clk)
    positions = []
    for _ in range(int(3 / (1/60))):
        clk.t += 1/60; m.update(1/60); positions.append(m.gaze.copy())
    moves = sum(1 for a, b in zip(positions, positions[1:]) if np.linalg.norm(b - a) > 1e-6)
    holds = len(positions) - moves
    assert moves > 0 and holds > moves            # mostly still, punctuated by jumps
    assert max(np.linalg.norm(p) for p in positions) <= m.cfg.gaze_radius + 1e-6


def test_blink_closes_fast_opens_slower():
    clk = Clock(); m = EyeMotion(EyeMotionConfig(blink_gap_s=0.0, blink_close_s=(0.05, 0.05)), rng=np.random.default_rng(0), clock=clk)
    closing = opening = 0
    for _ in range(int(3 / 0.005)):
        clk.t += 0.005; before = m.blink; m.update(0.005)
        if m.blink > before: closing += 1
        elif m.blink < before: opening += 1
    assert closing > 0 and opening > closing * 1.5


def test_pupil_reacts_to_emotion():
    clk = Clock(); m = EyeMotion(rng=np.random.default_rng(0), clock=clk)
    run(m, clk, 2.0, emotion=Emotion.SURPRISE); wide = m.pupil
    run(m, clk, 2.0, emotion=Emotion.ANGRY); narrow = m.pupil
    assert wide > m.cfg.pupil_base * 1.15 and narrow < m.cfg.pupil_base * 0.8


def test_look_at_and_lid_tracking():
    clk = Clock(); m = EyeMotion(EyeMotionConfig(micro=0.0), rng=np.random.default_rng(0), clock=clk)
    m.look_at(0.0, 1.0)
    run(m, clk, 1.0)
    assert m.gaze[1] > 0.8 * m.cfg.gaze_radius
    assert m.upper_lid_extra > 0.25


@pytest.mark.skipif(not os.path.isdir(GOAT), reason="goat eye parts missing")
def test_compositor_renders_and_reacts():
    eye = TexturedEye(TexturedEyeAssets(GOAT, size=96), lid_open=0.55)
    open_frame = eye.render((0.0, 0.0), 0.2, 0.0)
    assert open_frame.shape == (96, 96, 4) and open_frame.dtype == np.uint8
    assert open_frame[..., 3].mean() > 20                     # something visible
    closed = eye.render((0.0, 0.0), 0.2, 1.0)
    assert closed[..., 3].sum() < open_frame[..., 3].sum() * 0.1   # blink hides the eye
    small = eye.render((0.0, 0.0), 0.12, 0.0); big = eye.render((0.0, 0.0), 0.36, 0.0)
    dark = lambda f: ((f[..., :3].sum(axis=2) < 30) & (f[..., 3] > 128)).sum()
    assert dark(big) > dark(small)                            # dilation grows the pupil
    left = eye.render((-0.3, 0.0), 0.2, 0.0); right = eye.render((0.3, 0.0), 0.2, 0.0)
    cx = lambda f: np.average(np.arange(96), weights=((f[..., :3].sum(axis=2) < 30) & (f[..., 3] > 128)).sum(axis=0) + 1e-6)
    assert cx(right) > cx(left) + 5                           # pupil moved with the gaze
    assert eye.render((0.3, 0.0), 0.2, 0.0) is right           # cached for identical state


@pytest.mark.parametrize("scale", [0.7, 1.0, 1.2])
def test_blink_rate_is_sane_under_emotion_multiplier(scale):
    clk = Clock(); m = EyeMotion(EyeMotionConfig(blink_gap_s=2.0), rng=np.random.default_rng(3), clock=clk)
    blinks = 0; was_open = True
    for _ in range(int(30 / 0.01)):
        clk.t += 0.01; m.update(0.01, Emotion.NEUTRAL, blink_scale=scale)
        if was_open and m.blink > 0.5:
            blinks += 1; was_open = False
        elif m.blink < 0.1:
            was_open = True
    # 30 s at a 0-2 s gap: roughly 10-30 blinks scaled; never hundreds, never zero
    assert 5 <= blinks <= 60, blinks


def test_eye_lids_mask_hides_more_as_lids_close():
    """The lid mask for image eyes (EVE) must hide progressively more of the eye."""
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init(); pygame.display.set_mode((1, 1), pygame.HIDDEN)
    from talker.face_asset_loader import AssetFaceRenderer, FaceAssetLoader, default_manifest
    m = default_manifest("lids"); m.eye_lids = True
    r = AssetFaceRenderer(FaceAssetLoader().build(m))
    visible = []
    for upper, lower in ((0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.6, 0.4)):
        mask = r._lid_mask(120, 60, upper, lower, 0.0, 0.0, False, (0.1, 0.9))
        visible.append(120 * 60 if mask is None else int(pygame.surfarray.pixels_alpha(mask).astype(int).sum() / 255))
    assert visible[0] > visible[1] > visible[2] > visible[3]


def test_inner_glow_is_crisp_and_lit_from_within():
    """glow_style 'inner': nothing drawn outside the polygon, brighter core than edge."""
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init(); pygame.display.set_mode((1, 1), pygame.HIDDEN)
    from talker.face_asset_loader import _lit_polygon
    pts = [(60, 0), (0, 100), (120, 100)]
    sprite, (x0, y0) = _lit_polygon(pts, (255, 160, 0), (255, 255, 230), None, 0.15)
    assert (x0, y0) == (-1, -1)
    alpha = pygame.surfarray.pixels_alpha(sprite)
    assert alpha[2, 2] == 0                       # outside the triangle: transparent, no halo
    assert alpha[60 - x0, 90 - y0] == 255         # inside: opaque
    core = sprite.get_at((60 - x0, 75 - y0))
    edge = sprite.get_at((8 - x0, 98 - y0))
    assert sum(core[:3]) > sum(edge[:3]) + 100    # lit from within
