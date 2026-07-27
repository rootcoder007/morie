"""Tests for msmphr.msm_proportional_hazards."""

import numpy as np
import pytest

from morie.fn.msmphr import msm_proportional_hazards


def test_msmphr_basic():
    rng = np.random.default_rng(42)
    n = 4000
    c = rng.normal(size=n)
    A = (rng.random(n) < 1 / (1 + np.exp(-1.2 * c))).astype(float)
    t_event = rng.exponential(np.exp(-(0.5 * A + 0.8 * c)))
    cens = rng.exponential(3.0, size=n)
    time = np.minimum(t_event, cens)
    event = (t_event <= cens).astype(float)
    out = msm_proportional_hazards(time, event, A, c)
    assert abs(out["log_hr"] - 0.5) < abs(out["log_hr_unweighted"] - 0.5)
    assert out["hazard_ratio"] == pytest.approx(np.exp(out["log_hr"]))


def test_msmphr_edge():
    with pytest.raises(ValueError):
        msm_proportional_hazards(np.ones(20), np.zeros(20), np.zeros(20), np.zeros(20))  # no events
    with pytest.raises(ValueError):
        msm_proportional_hazards(-np.ones(20), np.ones(20), np.zeros(20), np.zeros(20))
