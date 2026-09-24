"""Tests for fastm.fast_mcd."""

from morie.fn import _array_core as np

import math

import pytest

from morie.fn.fastm import fast_mcd


def test_fastm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    result = fast_mcd(X, n_starts=10)
    assert isinstance(result, dict)
    # all keys promised by the docstring must be present
    expected_keys = {"estimate", "center", "cov_raw", "cov", "factor",
                     "subset", "h", "n", "p", "n_starts_used", "dets"}
    assert expected_keys.issubset(result.keys())
    # the determinant (estimate) should be a finite number
    assert math.isfinite(result["estimate"])
    # dimensions must match the input data
    n, p = X.shape
    assert result["n"] == n
    assert result["p"] == p
    assert len(result["center"]) == p
    assert len(result["cov"]) == p
    for row in result["cov"]:
        assert len(row) == p


def test_fastm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10, 3))
    # h must be greater than p
    with pytest.raises(ValueError):
        fast_mcd(X, h=2)
    # h cannot exceed the number of observations
    with pytest.raises(ValueError):
        fast_mcd(X, h=11)
    # need more than p+1 observations
    X_small = rng.normal(0, 1, (2, 3))
    with pytest.raises(ValueError):
        fast_mcd(X_small)
