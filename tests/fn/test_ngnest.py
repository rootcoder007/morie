"""Tests for ngnest. Full anchor: wave3/anchor_intermittent.py."""
import math
import pytest
from morie.fn.ngnest import (aggregate_forecasts, ensemble_members,
                             nbeats_ensemble)

SIG = [10.0 + 0.3 * t + 2.0 * math.sin(2 * math.pi * t / 12.0)
       for t in range(96)]


def test_the_ensemble_spans_lookbacks_and_block_sets():
    e = nbeats_ensemble(SIG, 12)
    assert e["n_members"] >= 6
    assert len(e["lookbacks"]) >= 3
    assert e["lookback_spread"] > 0.0


def test_the_median_absorbs_a_diverging_member_and_the_mean_does_not():
    """Which is exactly why the paper aggregates by median."""
    mem = ensemble_members(SIG, 12)
    base_med = aggregate_forecasts(mem, how="median")
    base_mean = aggregate_forecasts(mem, how="mean")
    poisoned = list(mem) + [{"lookback": 24, "multiple": 2,
                             "block_set": 0, "forecast": [1e6] * 12,
                             "residual_norm": 0.0}]
    assert abs(aggregate_forecasts(poisoned, "median")[0]
               - base_med[0]) < 1.0
    assert abs(aggregate_forecasts(poisoned, "mean")[0]
               - base_mean[0]) > 1000.0


def test_argument_checks():
    mem = ensemble_members(SIG, 12)
    with pytest.raises(ValueError):
        aggregate_forecasts(mem, how="nope")
    with pytest.raises(ValueError):
        aggregate_forecasts([])
    with pytest.raises(ValueError):
        aggregate_forecasts([{"forecast": [1.0]},
                             {"forecast": [1.0, 2.0]}])
    with pytest.raises(ValueError):
        ensemble_members([1.0, 2.0, 3.0], 12)
