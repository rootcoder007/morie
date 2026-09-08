"""cslnc: cosine learning-rate schedule with warmup (Loshchilov & Hutter 2017)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.cslnc import cosine_lr_schedule as cos


def _v(step, **kw):
    return float(np.asarray(cos(step, **kw)["value"]).ravel()[0])


def test_cslnc_warmup_is_linear_from_zero_to_lr_max():
    """During warmup the rate rises linearly; at the end of warmup it is
    exactly lr_max."""
    kw = dict(lr_max=1.0, lr_min=0.0, total_steps=100, warmup_steps=10)
    assert _v(0, **kw) == pytest.approx(0.0, abs=1e-12)
    assert _v(5, **kw) == pytest.approx(0.5, rel=1e-9)
    assert _v(10, **kw) == pytest.approx(1.0, rel=1e-9)


def test_cslnc_decays_to_lr_min_at_the_final_step():
    kw = dict(lr_max=1.0, lr_min=0.1, total_steps=100, warmup_steps=10)
    assert _v(100, **kw) == pytest.approx(0.1, abs=1e-9)


def test_cslnc_is_monotone_decreasing_after_warmup():
    kw = dict(lr_max=1.0, lr_min=0.0, total_steps=100, warmup_steps=10)
    vals = [_v(s, **kw) for s in range(10, 101, 5)]
    assert vals == sorted(vals, reverse=True)


def test_cslnc_halfway_through_decay_is_the_midpoint():
    """cos(pi/2) = 0, so the cosine schedule sits exactly at
    lr_min + (lr_max - lr_min)/2 at the midpoint of the decay phase."""
    kw = dict(lr_max=1.0, lr_min=0.0, total_steps=110, warmup_steps=10)
    assert _v(60, **kw) == pytest.approx(0.5, abs=1e-9)


def test_cslnc_never_leaves_the_lr_min_lr_max_band():
    kw = dict(lr_max=0.9, lr_min=0.05, total_steps=200, warmup_steps=20)
    for s in range(0, 201):
        assert -1e-12 <= _v(s, **kw) <= 0.9 + 1e-12


def test_cslnc_zero_warmup_starts_at_lr_max():
    kw = dict(lr_max=1.0, lr_min=0.0, total_steps=50, warmup_steps=0)
    assert _v(0, **kw) == pytest.approx(1.0, rel=1e-9)


def test_cslnc_past_the_end_stays_at_lr_min():
    """A schedule queried beyond total_steps must clamp, not keep going
    negative or turn back up with the cosine."""
    kw = dict(lr_max=1.0, lr_min=0.2, total_steps=50, warmup_steps=5)
    assert _v(200, **kw) == pytest.approx(0.2, abs=1e-9)
