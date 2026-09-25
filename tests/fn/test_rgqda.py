"""Tests for bsaclass.rangayyan_qda (eq. 10.73)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_qda


X = [[0, 0], [1, 0.2], [0.3, 1], [1.2, 1.1], [0.5, 0.4],
     [3, 3], [4, 3.2], [3.3, 4.1], [4.2, 4.5], [3.9, 3.1]]
Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def _g(q, pts, prior):
    """ln P - 1/2 ln|C| - 1/2 (x - m)' C^-1 (x - m), C the unbiased
    class covariance (eqs. 10.68-10.69)."""
    n = len(pts)
    m = [sum(p[j] for p in pts) / n for j in range(2)]
    C = [[sum((p[a] - m[a]) * (p[b] - m[b]) for p in pts) / (n - 1) for b in range(2)] for a in range(2)]
    det = C[0][0] * C[1][1] - C[0][1] ** 2
    Ci = [[C[1][1] / det, -C[0][1] / det], [-C[1][0] / det, C[0][0] / det]]
    d = [q[0] - m[0], q[1] - m[1]]
    return math.log(prior) - 0.5 * math.log(det) - 0.5 * sum(Ci[a][b] * d[a] * d[b] for a in range(2) for b in range(2))


def test_rgqda_basic():
    """Discriminant values per class equal the recomputed quadratic form
    and the query goes to the larger one."""
    for q in ([1.0, 1.0], [3.5, 3.5], [2.2, 2.0]):
        r = rangayyan_qda(X, Y, q)
        g = [_g(q, [p for p, t in zip(X, Y) if t == k], 0.5) for k in (0, 1)]
        assert r["g"] == pytest.approx(g, abs=1e-12)
        assert r["assigned"] == (0 if g[0] >= g[1] else 1)


def test_rgqda_edge():
    """Priors shift the log term; a class with no more samples than
    features raises."""
    r = rangayyan_qda(X, Y, [2.2, 2.0], priors=[0.9, 0.1])
    g = [_g([2.2, 2.0], [p for p, t in zip(X, Y) if t == k], pr) for k, pr in ((0, 0.9), (1, 0.1))]
    assert r["g"] == pytest.approx(g, abs=1e-12)
    with pytest.raises(ValueError):
        rangayyan_qda(X[:6], Y[:5] + [1], [1.0, 1.0])
