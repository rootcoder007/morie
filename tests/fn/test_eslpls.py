"""Tests for eslpls.esl_pls."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.eslpls import esl_pls


def test_eslpls_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n, p = 40, 3
    X = rng_x.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    M = 2
    result = esl_pls(X, y, M)
    assert isinstance(result, dict)
    expected_keys = {
        "estimate",
        "beta",
        "intercept",
        "y_variance_explained",
        "M",
        "n",
        "p",
        "method",
    }
    for key in expected_keys:
        assert key in result
    assert result["M"] == M
    assert result["n"] == n
    assert result["p"] == p
    assert len(result["beta"]) == p
    assert math.isfinite(result["y_variance_explained"])
    assert math.isfinite(result["intercept"])
    assert math.isfinite(result["estimate"])


def test_eslpls_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    n, p = 40, 3
    X = rng_x.normal(0, 1, (n, p))
    y = rng_y.normal(0, 1, n)
    with pytest.raises(ValueError):
        esl_pls(X, y, 0)
    with pytest.raises(ValueError):
        esl_pls(X, y, min(n - 1, p) + 1)
