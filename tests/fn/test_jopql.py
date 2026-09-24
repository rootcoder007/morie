"""Tests for jopql.joseph_pinball_quantile_loss."""

import math

from morie.fn import _array_core as np
from morie.fn.jopql import joseph_pinball_quantile_loss


def test_jopql_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = joseph_pinball_quantile_loss(y, q, tau)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "total" in result
    assert "coverage" in result
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["total"])
    assert 0.0 <= result["coverage"] <= 1.0


def test_jopql_edge():
    """Test edge cases."""
    # When y equals q exactly, pinball loss is zero at any quantile level
    y = [0.5, 1.5, 2.5, 3.5, 4.5]
    q = [0.5, 1.5, 2.5, 3.5, 4.5]
    tau = 0.5
    result = joseph_pinball_quantile_loss(y, q, tau)
    assert isinstance(result, dict)
    assert "loss" in result
    assert math.isclose(result["loss"], 0.0, abs_tol=1e-12)
