"""Tests for nie.natural_indirect_effect."""

from morie.fn import _array_core as np
from morie.fn.nie import natural_indirect_effect


def test_nie_basic():
    """Test basic functionality."""
    y11 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = natural_indirect_effect(y11, y10)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_nie_edge():
    """Test edge cases."""
    y11 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = natural_indirect_effect(y11, y10)
    assert isinstance(result, dict)


def test_nie_is_the_paired_contrast():
    import math

    import pytest

    y11 = [3.0, 4.5, 2.0, 5.0]
    y10 = [2.5, 3.0, 2.2, 4.1]
    d = [a - b for a, b in zip(y11, y10)]
    m = sum(d) / 4
    r = natural_indirect_effect(y11, y10)
    assert r["estimate"] == pytest.approx(m, rel=1e-14)
    assert r["se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in d) / 3 / 4), rel=1e-13)
