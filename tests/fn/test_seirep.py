"""Tests for seirep.seir_compartmental."""

import math

from morie.fn.seirep import seir_compartmental


def _get_estimate(result):
    if isinstance(result, dict):
        return result["estimate"]
    return result.estimate


def test_seirep_basic():
    """Test basic functionality."""
    S, E, I_, R = 999.0, 0.0, 1.0, 0.0
    beta, sigma, gamma = 0.5, 1.0 / 3.0, 1.0 / 7.0
    result = seir_compartmental(S, E, I_, R, beta, sigma, gamma)
    est = _get_estimate(result)
    assert math.isfinite(est)
    N = S + E + I_ + R
    assert 0.0 <= est <= N


def test_seirep_edge():
    """Test edge cases."""
    S, E, I_, R = 999.0, 0.0, 1.0, 0.0
    beta, sigma, gamma = 0.5, 1.0 / 3.0, 1.0 / 7.0
    result = seir_compartmental(S, E, I_, R, beta, sigma, gamma, t_max=0.0)
    est = _get_estimate(result)
    assert math.isfinite(est)
    assert est == 0.0
