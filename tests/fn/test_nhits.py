"""Tests for nhits. Full anchor: wave3/anchor_intermittent.py."""
import math
import pytest
from morie.fn import _s03core as k
from morie.fn.nhits import (expressiveness_knots, linear_interpolate,
                            max_pool, nhits_forecast, nhits_stack)

SIG = [math.sin(2 * math.pi * t / 24.0)
       + 0.6 * math.sin(2 * math.pi * t / 3.0) for t in range(96)]


def test_pooling_destroys_the_high_frequency_component():
    """That is the mechanism -- a pooled block can only fit slow
    components. Measured as what SURVIVES pooling: comparing
    step-to-step differences across the raw and pooled series would
    compare different sampling rates."""
    fast = [0.6 * math.sin(2 * math.pi * t / 3.0) for t in range(96)]
    slow = [math.sin(2 * math.pi * t / 24.0) for t in range(96)]
    fast_keep = k.sd(max_pool(fast, 6)) / k.sd(fast)
    slow_keep = k.sd(max_pool(slow, 6)) / k.sd(slow)
    assert fast_keep < 0.25
    assert slow_keep > 0.7
    assert slow_keep > 3.0 * fast_keep
    assert max_pool(SIG, 1) == pytest.approx(SIG)
    with pytest.raises(ValueError):
        max_pool(SIG, 0)
    with pytest.raises(ValueError):
        max_pool(SIG, 200)


def test_the_knot_count_is_ceil_r_times_h():
    assert expressiveness_knots(24, 0.25) == 6
    assert expressiveness_knots(24, 1.0) == 24
    assert expressiveness_knots(4, 0.01) == 2      # floor of 2
    with pytest.raises(ValueError):
        expressiveness_knots(10, 0.0)
    with pytest.raises(ValueError):
        expressiveness_knots(10, 1.5)


def test_interpolation_is_exact_at_the_knots():
    """Or the block is not predicting what it appears to be."""
    kn = [1.0, 4.0, 2.0, 8.0]
    out = linear_interpolate(kn, 10)
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(8.0)
    assert out[3] == pytest.approx(4.0)
    assert len(out) == 10
    assert len(linear_interpolate(kn, 1)) == 1
    with pytest.raises(ValueError):
        linear_interpolate([1.0], 5)


def test_coarse_blocks_predict_fewer_knots():
    fc, resid, tr = nhits_stack(SIG[:72], 24,
                                [(6, 0.25, 2), (2, 0.5, 2),
                                 (1, 1.0, 2)])
    assert tr[0]["n_knots"] < tr[-1]["n_knots"]
    assert tr[-1]["residual_norm"] < tr[0]["residual_norm"]
    r = nhits_forecast(SIG, 24, lookback=72)
    assert r["total_knots"] < r["dense_parameters"]
    with pytest.raises(ValueError):
        nhits_forecast(SIG, 24, lookback=4)
