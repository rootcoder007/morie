"""Tests for beta_standardized.beta_standardized."""

from morie.fn import _array_core as np

from morie.fn.beta_standardized import beta_standardized


def test_ca2e20_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b = 2.5
    s_x = 3.0
    s_y = 1.5
    result = beta_standardized(b, s_x, s_y)
    assert isinstance(result, dict)
    assert "value" in result
    expected = b * (s_x / s_y)
    assert result["value"] == expected
    assert "method" in result


def test_ca2e20_edge():
    """Test edge cases."""
    b = 0.0
    s_x = 5.0
    s_y = 2.0
    result = beta_standardized(b, s_x, s_y)
    assert isinstance(result, dict)
    assert result["value"] == 0.0
