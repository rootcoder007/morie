"""_sci_core checked against closed-form values on its used surfaces.

expit/logit are checked against the defining formula computed with the
stdlib; the distance tests use small integer matrices whose distances
are exact by hand (3-4-5 triangle etc.); the optimizer tests assert the
known analytic minima.
"""
import math

import pytest

from morie.fn import _sci_core as sc


def test_expit_logit():
    for v in (-5.0, -0.3, 0.0, 2.5):
        want = 1.0 / (1.0 + math.exp(-v))
        assert sc.expit(v) == pytest.approx(want, rel=1e-14)
    assert sc.logit(sc.expit(1.7)) == pytest.approx(1.7, rel=1e-12)


def test_cdist_pdist_squareform():
    a = [[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]]
    b = [[1.0, 0.0], [0.0, 2.0]]
    # euclidean distances, exact by hand
    want = [[1.0, 2.0],
            [math.sqrt(4 + 16), math.sqrt(9 + 4)],
            [1.0, math.sqrt(2)]]
    got = sc.cdist(a, b).tolist()
    for r1, r2 in zip(got, want):
        assert r1 == pytest.approx(r2, rel=1e-12)

    c = [[1.0, 2.0], [3.0, 4.0], [1.0, -1.0]]   # nonzero rows for cosine
    def cos_d(u, v):
        num = sum(x * y for x, y in zip(u, v))
        den = math.sqrt(sum(x * x for x in u)) * \
            math.sqrt(sum(y * y for y in v))
        return 1.0 - num / den
    pair = [(0, 1), (0, 2), (1, 2)]
    want_by_metric = {
        "euclidean": [math.sqrt(8), 3.0, math.sqrt(29)],
        "sqeuclidean": [8.0, 9.0, 29.0],
        "cityblock": [4.0, 3.0, 7.0],
        "chebyshev": [2.0, 3.0, 5.0],
        "cosine": [cos_d(c[i], c[j]) for i, j in pair],
    }
    for metric, want_v in want_by_metric.items():
        g = sc.pdist(c, metric=metric).tolist()
        assert g == pytest.approx(want_v, rel=1e-12)

    # squareform round-trip: vector -> symmetric matrix -> vector
    p = sc.pdist(a).tolist()
    sq = sc.squareform(sc.pdist(a))
    n = len(a)
    k = 0
    for i in range(n):
        assert sq.tolist()[i][i] == 0.0
        for j in range(i + 1, n):
            assert sq.tolist()[i][j] == pytest.approx(p[k], rel=1e-12)
            assert sq.tolist()[j][i] == pytest.approx(p[k], rel=1e-12)
            k += 1
    back = sc.squareform(sq)
    assert back.tolist() == pytest.approx(p, rel=1e-12)


def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def test_minimize_nelder_mead_rosenbrock():
    got = sc.minimize(rosen, [-1.2, 1.0], method="Nelder-Mead",
                      options={"maxiter": 5000})
    # analytic minimum: f(1, 1) = 0
    assert got.fun == pytest.approx(0.0, abs=1e-8)
    assert list(got.x) == pytest.approx([1.0, 1.0], abs=1e-3)


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
