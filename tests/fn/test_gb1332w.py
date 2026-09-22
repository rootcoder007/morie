"""Tests for gb1332w.gibbons_wrs_efficacy."""

from morie.fn import _array_core as np

from morie.fn.gb1332w import gibbons_wrs_efficacy


def test_gb1332w_basic():
    """Test basic functionality."""
    m = 5
    n = 7
    integral = 0.25
    result = gibbons_wrs_efficacy(m, n, integral)
    assert isinstance(result, dict)
    assert "efficacy" in result
    expected = 12.0 * m * n * integral * integral / (m + n + 1.0)
    assert result["efficacy"] == expected
    assert result["integral"] == integral
    assert result["m"] == m
    assert result["n"] == n
    assert result["method"] == "Mann-Whitney / rank-sum efficacy, eq. (13.3.10)"


def test_gb1332w_edge():
    """Test edge cases."""
    m = 1
    n = 1
    integral = 1.0
    result = gibbons_wrs_efficacy(m, n, integral)
    assert isinstance(result, dict)
    assert "efficacy" in result
    expected = 12.0 * m * n * integral * integral / (m + n + 1.0)
    assert result["efficacy"] == expected
