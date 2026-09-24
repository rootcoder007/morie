"""Tests for genvxt.generalizability_theory."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.genvxt import generalizability_theory


def test_genvxt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 5))
    facets = 5
    result = generalizability_theory(X, facets)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert "e_rho2" in result
    assert "phi" in result
    assert "var_p" in result
    assert "var_i" in result
    assert "var_pi" in result
    assert "ms_p" in result
    assert "ms_i" in result
    assert "ms_pi" in result
    assert "n_p" in result
    assert "n_i" in result
    assert "method" in result
    assert result["n_p"] == 40
    assert result["n_i"] == facets


def test_genvxt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    # Omit facets -> defaults to observed n_i
    result = generalizability_theory(X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n_p"] == 40
    assert result["n_i"] == 3
