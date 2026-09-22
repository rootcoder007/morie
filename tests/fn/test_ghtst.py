"""Tests for ghtst.ghosal_np_testing."""

from morie.fn import _array_core as np

from morie.fn.ghtst import ghosal_np_testing


def test_ghtst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    # Use depth=1 so each level has a single bin-pair (single-element arrays),
    # which is compatible with this numpy-shim's strict float conversion.
    result = ghosal_np_testing(x, depth=1)
    assert "statistic" in result
    assert "p_value" in result
    assert "BF10" in result
    assert "log_BF10" in result
    assert 0 <= result["p_value"] <= 1
    # Independent check of BF10 == exp(log_BF10):
    import math
    assert math.isclose(result["BF10"], math.exp(result["log_BF10"]), rel_tol=1e-9)


def test_ghtst_edge():
    """Test edge cases."""
    result = ghosal_np_testing(np.array([1.0]))
    assert result["n"] == 1
    assert result["p_value"] != result["p_value"]  # NaN check
