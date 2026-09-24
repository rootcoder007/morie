"""Tests for ctomeg.omega_total."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.ctomeg import omega_total


def test_ctomeg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 5
    X = rng.normal(0, 1, (n, p))
    factor_loadings = rng.normal(0, 1, p)
    result = omega_total(X, factor_loadings)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "omega" in result
    assert "alpha" in result
    assert "var_total" in result
    assert "uniquenesses" in result
    assert "p" in result
    assert result["p"] == p
    assert len(result["uniquenesses"]) == p
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["omega"])
    assert math.isfinite(result["alpha"])


def test_ctomeg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    factor_loadings = rng.normal(0, 1, p)
    result = omega_total(X, factor_loadings)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "omega" in result
    assert "alpha" in result
    assert "var_total" in result
    assert "uniquenesses" in result
    assert "p" in result
    assert result["p"] == p
    assert len(result["uniquenesses"]) == p
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["omega"])
    assert math.isfinite(result["alpha"])
