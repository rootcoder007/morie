"""Tests for magsd.ma_glass_delta."""

from morie.fn import _array_core as np

from morie.fn.magsd import ma_glass_delta


def test_magsd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m1 = float(rng.normal(5.0, 0.5))
    m2 = float(rng.normal(4.0, 0.5))
    s_ctrl = float(abs(rng.normal(1.0, 0.1)) + 0.5)
    n1 = int(rng.integers(10, 50))
    n2 = int(rng.integers(10, 50))
    while n2 < 2:
        n2 = int(rng.integers(10, 50))
    result = ma_glass_delta(m1, m2, s_ctrl, n1, n2)
    assert isinstance(result, dict)
    for key in ("delta", "var", "se", "ci_lo", "ci_hi", "n1", "n2"):
        assert key in result


def test_magsd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    m1 = float(rng.normal(0.0, 1.0))
    m2 = float(rng.normal(0.0, 1.0))
    s_ctrl = float(abs(rng.normal(1.0, 0.1)) + 0.5)
    n1 = 1
    n2 = 2
    result = ma_glass_delta(m1, m2, s_ctrl, n1, n2)
    assert isinstance(result, dict)
    assert "delta" in result
    assert result["n1"] == 1.0
    assert result["n2"] == 2.0
