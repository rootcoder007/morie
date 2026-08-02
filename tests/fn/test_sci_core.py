"""Equivalence: _sci_core vs scipy on the used surfaces."""

import math

import pytest

scipy_opt = pytest.importorskip("scipy.optimize")
from scipy.spatial import distance as sp_dist  # noqa: E402
from scipy.special import expit as sp_expit  # noqa: E402

from morie.fn import _sci_core as sc  # noqa: E402


def test_expit_logit():
    for v in (-5.0, -0.3, 0.0, 2.5):
        assert sc.expit(v) == pytest.approx(float(sp_expit(v)), rel=1e-14)
    assert sc.logit(sc.expit(1.7)) == pytest.approx(1.7, rel=1e-12)


def test_cdist_pdist_squareform():
    a = [[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]
    b = [[1.0, 0.0], [0.0, 2.0]]
    got = sc.cdist(a, b).tolist()
    want = sp_dist.cdist(a, b).tolist()
    for r1, r2 in zip(got, want):
        assert r1 == pytest.approx(r2, rel=1e-12)
    c = [[1.0, 2.0], [3.0, 4.0], [1.0, -1.0]]   # nonzero rows for cosine
    for metric in ("euclidean", "sqeuclidean", "cityblock", "chebyshev",
                   "cosine"):
        g = sc.pdist(c, metric=metric).tolist()
        w = sp_dist.pdist(c, metric=metric).tolist()
        assert g == pytest.approx(w, rel=1e-12)
    sq = sc.squareform(sc.pdist(a))
    wq = sp_dist.squareform(sp_dist.pdist(a))
    for r1, r2 in zip(sq.tolist(), wq.tolist()):
        assert r1 == pytest.approx(r2, rel=1e-12)
    back = sc.squareform(sq)
    assert back.tolist() == pytest.approx(sc.pdist(a).tolist(), rel=1e-12)


def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def test_minimize_nelder_mead_rosenbrock():
    got = sc.minimize(rosen, [-1.2, 1.0], method="Nelder-Mead",
                      options={"maxiter": 5000})
    ref = scipy_opt.minimize(rosen, [-1.2, 1.0], method="Nelder-Mead")
    assert got.fun == pytest.approx(0.0, abs=1e-8)
    assert list(got.x) == pytest.approx([1.0, 1.0], abs=1e-3)
    assert ref.fun == pytest.approx(got.fun, abs=1e-6)


def test_minimize_bfgs_quadratic():
    fun = lambda x: (x[0] - 3.0) ** 2 + 2 * (x[1] + 1.0) ** 2 + 5.0  # noqa: E731
    got = sc.minimize(fun, [0.0, 0.0], method="BFGS")
    assert got.fun == pytest.approx(5.0, abs=1e-8)
    assert list(got.x) == pytest.approx([3.0, -1.0], abs=1e-4)


def test_minimize_scalar():
    got = sc.minimize_scalar(lambda t: (t - 2.5) ** 2 + 1.0,
                             bounds=(0.0, 10.0))
    assert got.x == pytest.approx(2.5, abs=1e-6)
    assert got.fun == pytest.approx(1.0, abs=1e-10)
