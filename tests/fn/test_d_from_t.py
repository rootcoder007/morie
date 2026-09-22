"""Tests for d_from_t.d_from_t."""

from morie.fn import _array_core as np

from morie.fn.d_from_t import d_from_t


def test_ca11e5_basic():
    """Test basic functionality."""
    t = 2.0
    n1 = 50
    n2 = 50
    result = d_from_t(t, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    # Verify against the documented formula: d = t * sqrt((n1 + n2) / (n1 * n2))
    expected = t * np.sqrt((n1 + n2) / (n1 * n2))
    assert np.isclose(result["value"], expected)


def test_ca11e5_edge():
    """Test edge cases: equal group sizes, large t."""
    t = 5.0
    n1 = 30
    n2 = 70
    result = d_from_t(t, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent recomputation of the formula
    expected = t * np.sqrt((n1 + n2) / (n1 * n2))
    assert np.isclose(result["value"], expected)
