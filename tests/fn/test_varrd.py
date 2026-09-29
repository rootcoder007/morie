"""Tests for varrd.variance_reduction_split."""

from morie.fn import _array_core as np
from morie.fn.varrd import variance_reduction_split


def test_varrd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    split_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_reduction_split(y, split_idx)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_varrd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    split_idx = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_reduction_split(y, split_idx)
    assert isinstance(result, dict)


def test_variance_reduction_and_weighted_sse():
    import pytest

    y = [1.0, 2.0, 1.5, 6.0, 7.0, 6.5, 5.5]
    left = [1.0, 2.0, 1.5]
    right = [6.0, 7.0, 6.5, 5.5]

    def pv(v):
        m = sum(v) / len(v)
        return sum((t - m) ** 2 for t in v) / len(v)

    dv = pv(y) - 3 / 7 * pv(left) - 4 / 7 * pv(right)
    r = variance_reduction_split(y, [0, 1, 2])
    assert r["delta_var"] == pytest.approx(dv, rel=1e-13)
    assert r["sse_weighted"] == pytest.approx(pv(left) * 3 * 3 / 7 + pv(right) * 4 * 4 / 7, rel=1e-13)
    mask = [True, True, True, False, False, False, False]
    assert variance_reduction_split(y, mask)["delta_var"] == pytest.approx(dv, rel=1e-13)
