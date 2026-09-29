"""Tests for ebayes.empirical_bayes_shrinkage."""

from morie.fn import _array_core as np
from morie.fn.ebayes import empirical_bayes_shrinkage


def test_ebayes_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_shrinkage(y, cluster)
    assert isinstance(result, dict)
    assert "clusters" in result


def test_ebayes_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_shrinkage(y, cluster)
    assert isinstance(result, dict)


def test_shrinkage_factors_recomputed():
    import pytest

    y = [3.0, 4.0, 5.0, 6.0, 8.0, 2.0, 2.5, 3.5, 3.0, 9.0, 7.0]
    cl = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]
    labs = [0, 1, 2, 3]
    groups = [[y[i] for i in range(11) if cl[i] == c] for c in labs]
    means = [sum(g) / len(g) for g in groups]
    nj = [len(g) for g in groups]
    grand = sum(m * k for m, k in zip(means, nj)) / sum(nj)
    s2e = sum((v - means[c]) ** 2 for c, g in enumerate(groups) for v in g) / (11 - 4)
    mm = sum(means) / 4
    between = sum((m - mm) ** 2 for m in means) / 3
    s2u = max(between - s2e / (sum(nj) / 4), 0.0)
    lam = [s2u / (s2u + s2e / k) for k in nj]
    r = empirical_bayes_shrinkage(y, cl)
    assert [float(v) for v in r["lambda"]] == pytest.approx(lam, rel=1e-12)
    assert [float(v) for v in r["shrunk"]] == pytest.approx(
        [l_ * m + (1 - l_) * grand for l_, m in zip(lam, means)], rel=1e-12
    )
