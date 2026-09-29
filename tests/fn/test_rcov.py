"""Tests for rcov.py - Riemannian covariance."""

from morie.fn import _array_core as np
from morie.fn.rcov import rcov, rcov_fn


def test_rcov_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = rcov_fn(X)
    assert result.name == "riemannian_cov"
    assert "covariance" in result.extra
    assert "metric" in result.extra


def test_rcov_symmetric():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = rcov_fn(X)
    C = result.extra["covariance"]
    assert np.allclose(C, C.T)


def test_rcov_logeuclid():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = rcov_fn(X, metric="logeuclid")
    assert result.extra["metric"] == "logeuclid"
    C = result.extra["covariance"]
    assert np.allclose(C, C.T)


def test_rcov_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2, 50))
    result = rcov(X)
    assert result.name == "riemannian_cov"


def test_covariance_and_log_euclidean_map():
    import math

    import pytest

    X = [[1.0, 2.0, 4.0, 3.0], [0.5, 1.0, 2.5, 1.0]]
    n = 4
    m = [sum(r) / n for r in X]
    C = [[sum((X[a][k] - m[a]) * (X[b][k] - m[b]) for k in range(n)) / (n - 1) for b in range(2)] for a in range(2)]
    r = rcov_fn(X)
    got = r.extra["covariance"]
    for a in range(2):
        assert [float(v) for v in got[a]] == pytest.approx(C[a], rel=1e-13)
    L = rcov_fn(X, metric="logeuclid").extra["covariance"]
    # log of an SPD matrix: its trace is log det
    det = C[0][0] * C[1][1] - C[0][1] ** 2
    assert float(L[0][0]) + float(L[1][1]) == pytest.approx(math.log(det), rel=1e-10)
