"""Tests for tmlnde.tmle_natural_direct."""

from morie.fn import _array_core as np
from morie.fn.tmlnde import tmle_natural_direct


def test_tmlnde_basic():
    """Test basic functionality."""
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y00 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_natural_direct(y10, y00)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlnde_edge():
    """Test edge cases."""
    y10 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y00 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = tmle_natural_direct(y10, y00)
    assert isinstance(result, dict)


def test_nde_is_the_paired_contrast():
    import math

    import pytest

    y10 = [3.0, 4.5, 2.0, 5.0]
    y00 = [2.5, 3.0, 2.2, 4.1]
    d = [a - b for a, b in zip(y10, y00)]
    m = sum(d) / 4
    r = tmle_natural_direct(y10, y00)
    assert r["estimate"] == pytest.approx(m, rel=1e-14)
    assert r["se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in d) / 3 / 4), rel=1e-13)
