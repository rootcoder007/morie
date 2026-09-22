"""Tests for aitpow.aitchison_powering."""

from morie.fn import _array_core as np

from morie.fn.aitpow import aitchison_powering


def test_aitpow_basic():
    """Test basic functionality."""
    alpha = 2.0
    x = np.array([0.1, 0.2, 0.3, 0.4])
    result = aitchison_powering(alpha, x)
    assert isinstance(result, dict)
    assert "composition" in result
    assert "a" in result
    assert "total" in result
    assert "D" in result
    assert result["a"] == 2.0
    assert result["D"] == 4
    assert result["total"] == 1.0

    p = [v ** alpha for v in x]
    s = sum(p)
    expected = [1.0 * v / s for v in p]
    assert len(result["composition"]) == len(x)
    for got, want in zip(result["composition"], expected):
        assert abs(got - want) < 1e-12
    assert abs(sum(result["composition"]) - 1.0) < 1e-12


def test_aitpow_edge():
    """Test edge cases."""
    x = np.array([1.0, 1.0, 1.0, 1.0])
    result = aitchison_powering(0.5, x)
    assert isinstance(result, dict)
    assert "composition" in result
    assert result["a"] == 0.5
    assert result["D"] == 4
    assert abs(sum(result["composition"]) - 1.0) < 1e-12
    for v in result["composition"]:
        assert abs(v - 0.25) < 1e-12
