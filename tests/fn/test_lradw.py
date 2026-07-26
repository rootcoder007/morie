"""lradw: linear learning-rate warmup (Vaswani et al. 2017)."""

import numpy as np
import pytest

from morie.fn.lradw import lr_warmup as warm


def _v(step, **kw):
    return float(np.asarray(warm(step, **kw)["value"]).ravel()[0])


def test_lradw_rises_linearly_to_the_target():
    kw = dict(lr_target=1.0, warmup_steps=10)
    assert _v(0, **kw) == pytest.approx(0.0, abs=1e-12)
    assert _v(5, **kw) == pytest.approx(0.5, rel=1e-9)
    assert _v(10, **kw) == pytest.approx(1.0, rel=1e-9)


def test_lradw_holds_the_target_after_warmup():
    kw = dict(lr_target=0.7, warmup_steps=10)
    for s in (10, 11, 50, 1000):
        assert _v(s, **kw) == pytest.approx(0.7, rel=1e-9)


def test_lradw_is_monotone_nondecreasing():
    kw = dict(lr_target=1.0, warmup_steps=25)
    vals = [_v(s, **kw) for s in range(0, 40)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_lradw_slope_is_lr_target_over_warmup_steps():
    """Each step adds exactly lr_target / warmup_steps."""
    kw = dict(lr_target=2.0, warmup_steps=8)
    step = 2.0 / 8
    for s in range(0, 8):
        assert _v(s + 1, **kw) - _v(s, **kw) == pytest.approx(step, rel=1e-9)


def test_lradw_rejects_a_non_positive_warmup_length():
    """A zero-length warmup is a division by zero, so it raises rather than
    silently degenerating. (cslnc allows warmup_steps=0 and means "no warmup";
    the two modules differ here on purpose and the contracts say so.)"""
    with pytest.raises(ValueError, match="warmup_steps must be > 0"):
        warm(0, lr_target=0.5, warmup_steps=0)


def test_lradw_starting_at_zero_is_the_point():
    """Warmup exists so the first optimiser steps are tiny; starting at the
    target would defeat it."""
    assert _v(0, lr_target=1.0, warmup_steps=100) < 0.02
