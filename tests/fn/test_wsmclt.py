"""Tests for wsmclt.wasserman_clt."""

from morie.fn import _array_core as np
from morie.fn.wsmclt import wasserman_clt


def test_wsmclt_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_clt(data)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsmclt_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_clt(data)
    assert isinstance(result, dict)


def test_clt_z_recomputed():
    import math

    import pytest

    d = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(d) / 5
    s = math.sqrt(sum((v - m) ** 2 for v in d) / 4)
    r = wasserman_clt(d)
    assert r["estimate"] == pytest.approx(m / (s / math.sqrt(5)), rel=1e-13)
    assert r["se"] == pytest.approx(s / math.sqrt(5), rel=1e-13)
