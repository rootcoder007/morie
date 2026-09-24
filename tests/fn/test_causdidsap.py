"""Tests for causdidsap.causal_did_sun_abraham."""

import math

from morie.fn import _array_core as np

from morie.fn.causdidsap import causal_did_sun_abraham


def test_causdidsap_basic():
    """Test basic functionality with never-treated control."""
    rng = np.random.default_rng(42)
    n, T = 40, 10
    Y_panel = rng.normal(0, 1, (n, T))
    G_first_treat = [3] * 30 + [math.inf] * 10
    result = causal_did_sun_abraham(
        Y_panel, G_first_treat, rel_periods=list(range(-3, 6)))
    assert isinstance(result, dict)
    assert "mu" in result
    assert "naive_twfe" in result
    assert "cohorts" in result
    assert "weights" in result
    assert "rel_periods" in result
    assert "n_units" in result
    assert "n_periods" in result
    assert len(result["mu"]) == 9
    assert len(result["naive_twfe"]) == 9
    assert len(result["cohorts"]) >= 1
    assert result["n_units"] == n
    assert result["n_periods"] == T


def test_causdidsap_edge():
    """Test with notyet control group and different periodisation."""
    rng = np.random.default_rng(7)
    n, T = 40, 10
    Y_panel = rng.normal(0, 1, (n, T))
    G_first_treat = [3] * 30 + [math.inf] * 10
    result = causal_did_sun_abraham(
        Y_panel,
        G_first_treat,
        rel_periods=list(range(-2, 4)),
        control="notyet")
    assert isinstance(result, dict)
    assert "mu" in result
    assert "naive_twfe" in result
    assert "catt" in result
    assert "weights" in result
    assert len(result["mu"]) == 6
    assert len(result["naive_twfe"]) == 6
    assert result["n_units"] == n
    assert result["n_periods"] == T
