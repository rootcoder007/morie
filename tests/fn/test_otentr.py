"""Tests for otentr.ot_entropy_regulariser."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.otentr import ot_entropy_regulariser


def test_otentr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    # Build a non-negative coupling matrix (4x4)
    T = rng.uniform(0.0, 1.0, (4, 4))
    epsilon = 1e-6
    result = ot_entropy_regulariser(T, epsilon)
    assert isinstance(result, dict)
    for key in ("estimate", "entropy", "epsilon", "n", "method"):
        assert key in result
    assert result["epsilon"] == epsilon
    assert result["n"] == 4 * 4
    # estimate = epsilon * entropy by construction
    assert result["estimate"] == pytest.approx(epsilon * result["entropy"])


def test_otentr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    # Smallest sensible non-trivial coupling: a 2x2 matrix
    T = rng.uniform(0.0, 1.0, (2, 2))
    epsilon = 1e-6
    result = ot_entropy_regulariser(T, epsilon)
    assert isinstance(result, dict)
    assert result["n"] == 4
    assert math.isfinite(result["entropy"])
    assert math.isfinite(result["estimate"])
    # entropy must be non-negative: -sum v*(log v - 1) = sum v*(1 - log v) >= 0
    assert result["entropy"] >= 0.0
