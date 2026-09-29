"""Tests for cluvar.cluster_variance."""

from morie.fn import _array_core as np
from morie.fn.cluvar import cluster_variance


def test_cluvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_variance(y, cluster)
    assert isinstance(result, dict)
    assert "mean" in result


def test_cluvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_variance(y, cluster)
    assert isinstance(result, dict)


def test_cluster_variance_and_design_effect_recomputed():
    import math

    import pytest

    y = [3.0, 4.0, 5.0, 6.0, 7.0, 5.0, 2.0, 2.5, 3.5, 5.0, 4.0, 6.0]
    cl = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    means = [sum(y[i] for i in range(12) if cl[i] == c) / 3 for c in range(4)]
    g = sum(means) / 4
    sb2 = sum((m - g) ** 2 for m in means) / 3
    var = (1 - 4 / 10) * sb2 / 4
    grand = sum(y) / 12
    ssb = sum(3 * (m - grand) ** 2 for m in means)
    ssw = sum((y[i] - means[cl[i]]) ** 2 for i in range(12))
    msb, msw = ssb / 3, ssw / 8
    icc = (msb - msw) / (msb + 2 * msw)
    r = cluster_variance(y, cl, N=10)
    assert r["variance"] == pytest.approx(var, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(var), rel=1e-13)
    assert r["icc"] == pytest.approx(icc, rel=1e-12)
    assert r["deff"] == pytest.approx(1 + 2 * icc, rel=1e-12)
