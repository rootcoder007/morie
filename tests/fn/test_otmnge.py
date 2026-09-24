"""Tests for otmnge.ot_marginal_negent."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.otmnge import ot_marginal_negent


def test_otmnge_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    T = rng.uniform(0, 1, (5, 3))
    result = ot_marginal_negent(T)
    assert isinstance(result, dict)
    for key in ("estimate", "shannon", "mass", "nrow", "ncol", "method"):
        assert key in result
    assert result["nrow"] == 5
    assert result["ncol"] == 3
    assert math.isfinite(result["estimate"])
    assert result["mass"] >= 0.0


def test_otmnge_edge():
    """Test edge cases."""
    T = np.zeros((2, 2))
    result = ot_marginal_negent(T)
    assert isinstance(result, dict)
    for key in ("estimate", "shannon", "mass", "nrow", "ncol", "method"):
        assert key in result
    assert result["nrow"] == 2
    assert result["ncol"] == 2
    assert math.isfinite(result["estimate"])
    assert result["mass"] == 0.0
