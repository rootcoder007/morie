"""Tests for causinst.causal_iv_instrumental_dag."""

import math

from morie.fn import _array_core as np

from morie.fn.causinst import causal_iv_instrumental_dag


def test_causinst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 40)
    D = np.random.default_rng(42).integers(0, 2, 40)
    Z = np.random.default_rng(43).integers(0, 2, 40)
    result = causal_iv_instrumental_dag(y, D, Z)
    # Check that the result contains the documented keys
    for key in ("beta", "se", "estimand", "relevance", "relevance_p",
                "assumptions", "testable", "untestable", "n", "method"):
        assert key in result
    # Numeric outputs should be finite
    assert math.isfinite(result["beta"])
    assert math.isfinite(result["se"])
    assert math.isfinite(result["relevance"])
    # Sample size matches input
    assert result["n"] == 40


def test_causinst_edge():
    """Test edge cases."""
    y = np.random.default_rng(0).normal(0, 1, 40)
    D = np.random.default_rng(1).integers(0, 2, 40)
    Z = np.random.default_rng(2).integers(0, 2, 40)
    result = causal_iv_instrumental_dag(y, D, Z, homogeneous=True)
    # When homogeneous=True the estimand should describe the ATE
    assert isinstance(result["estimand"], str)
    assert "average treatment effect" in result["estimand"]
