"""Tests for d_probit.d_probit."""

from morie.fn import _array_core as np

from morie.fn.d_probit import d_probit


def test_ca11e20_basic():
    """Test basic functionality."""
    result = d_probit(0.8, 0.2)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    assert result["value"] > 0


def test_ca11e20_edge():
    """Test edge cases."""
    result = d_probit(0.5, 0.5)
    assert isinstance(result, dict)
    import math
    assert math.isfinite(result["value"])
    assert math.isclose(result["value"], 0.0, abs_tol=1e-9)
