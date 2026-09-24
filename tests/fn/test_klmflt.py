"""Tests for klmflt.kalman_filter."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.klmflt import kalman_filter


def test_klmflt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 50
    y = rng.normal(0, 1, n)
    # Build simple scalar state space model
    model = {
        "F": [[0.9]],
        "H": [[1.0]],
        "Q": [[0.1]],
        "R": [[0.1]],
    }
    result = kalman_filter(y, model)
    # Result is a dict-like RichResult
    assert isinstance(result, dict)
    # Check that expected keys are present
    assert "estimate" in result
    assert "state" in result
    assert "loglik" in result
    assert "n" in result
    # Check that n matches input length
    assert result["n"] == n
    # Check that log-likelihood is a finite number
    assert math.isfinite(result["loglik"])


def test_klmflt_edge():
    """Test edge cases."""
    model = {
        "F": [[0.9]],
        "H": [[1.0]],
        "Q": [[0.1]],
        "R": [[0.1]],
    }
    # Empty y is invalid and should raise ValueError
    with pytest.raises(ValueError):
        kalman_filter([], model)
