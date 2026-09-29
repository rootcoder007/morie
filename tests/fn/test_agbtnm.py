"""Tests for agbtnm.alphazero_batch_norm."""

from morie.fn import _array_core as np
from morie.fn.agbtnm import alphazero_batch_norm


def test_agbtnm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_batch_norm(x)
    assert isinstance(result, dict)
    assert "runmean" in result


def test_agbtnm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_batch_norm(x)
    assert isinstance(result, dict)


def test_agbtnm_running_statistics_recomputed():
    import math

    import pytest

    x = [0.3, -1.2, 0.8, 2.0, 0.1]
    m = len(x)
    mu = sum(x) / m
    vb = sum((v - mu) ** 2 for v in x) / m
    rm = 0.8 * 0.5 + 0.2 * mu
    rv = 0.8 * 1.5 + 0.2 * vb * m / (m - 1)
    r = alphazero_batch_norm(x, runmean=0.5, runvar=1.5, momentum=0.2, eps=1e-3, gamma=2.0, beta=-1.0)
    assert r["runmean"] == pytest.approx(rm, rel=1e-14)
    assert r["runvar"] == pytest.approx(rv, rel=1e-14)
    assert r["normalized"] == pytest.approx([2.0 * (v - rm) / math.sqrt(rv + 1e-3) - 1.0 for v in x], rel=1e-13)
    assert r["trainnorm"] == pytest.approx([2.0 * (v - mu) / math.sqrt(vb + 1e-3) - 1.0 for v in x], rel=1e-13)
