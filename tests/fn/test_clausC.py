"""Tests for clausC.clausius_clapeyron."""

import math

from morie.fn import _array_core as np

from morie.fn.clausC import clausius_clapeyron


def test_clausC_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    T = rng.uniform(270.0, 310.0, 50)
    result = clausius_clapeyron(T)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "rate_percent_per_K" in result
    assert "es" in result
    assert "des_dt" in result
    assert "T" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 50
    assert len(result["T"]) == 50
    assert len(result["es"]) == 50
    assert len(result["des_dt"]) == 50
    assert np.mean(result["rate"]) == result["estimate"]


def test_clausC_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    T = np.linspace(260.0, 300.0, 10)
    result = clausius_clapeyron(T)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] > 0.0
    assert np.min(np.asarray(result["es"])) > 0.0
    assert np.min(np.asarray(result["des_dt"])) > 0.0
    assert result["n"] == 10
