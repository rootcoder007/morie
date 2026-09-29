"""Tests for cluster.one_stage_cluster."""

from morie.fn import _array_core as np
from morie.fn.cluster import one_stage_cluster


def test_cluster_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(42).normal(0, 1, 100)
    result = one_stage_cluster(Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cluster_edge():
    """Test edge cases."""
    Y = np.random.default_rng(42).normal(0, 1, 100)
    result = one_stage_cluster(Y)
    assert isinstance(result, dict)


def test_one_stage_cluster_estimator_recomputed():
    import math

    import pytest

    Y = [[3.0, 4.0, 5.0], [6.0, 7.0, 5.0], [2.0, 2.5, 3.5], [5.0, 4.0, 6.0]]
    m, k, M = 4, 3, 20
    cm = [sum(r) / k for r in Y]
    est = sum(cm) / m
    sb2 = sum((c - est) ** 2 for c in cm) / (m - 1)
    var = (M - m) / M * sb2 / m
    r = one_stage_cluster(Y, M=M, level=0.9)
    assert r["estimate"] == pytest.approx(est, rel=1e-14)
    assert r["se"] == pytest.approx(math.sqrt(var), rel=1e-13)
    flat = [v for row in Y for v in row]
    mf = sum(flat) / 12
    s2 = sum((v - mf) ** 2 for v in flat) / 11
    assert r["deff"] == pytest.approx(var / (s2 / 12), rel=1e-12)
