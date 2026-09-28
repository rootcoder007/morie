import math

import pytest

from morie.fn.fastm import fast_mcd
from morie.fn.robustda import robust_lda

X = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.4],
    [9, 0],
    [0.2, 0.8],
    [5, 5],
    [6, 5],
    [5, 6],
    [6, 6],
    [5.5, 5.4],
    [5.2, 5.8],
    [-9, 9],
]
y = [0] * 7 + [1] * 7


def test_pooled_recomputes():
    r = robust_lda(X, y)
    f0, f1 = fast_mcd(X[:7]), fast_mcd(X[7:])
    for a in range(2):
        for b in range(2):
            want = (f0.h * f0.cov[a][b] + f1.h * f1.cov[a][b]) / (f0.h + f1.h - 2)
            assert r.pooled_cov[a][b] == pytest.approx(want, abs=1e-12)
    assert r.centers[0] == pytest.approx(list(f0.center), abs=1e-12)


def test_scores_formula():
    r = robust_lda(X, y, newdata=[[2.0, 3.0]])
    W = r.pooled_cov
    det = W[0][0] * W[1][1] - W[0][1] * W[1][0]
    Wi = [[W[1][1] / det, -W[0][1] / det], [-W[1][0] / det, W[0][0] / det]]
    for k, m in enumerate(r.centers):
        c = [Wi[0][0] * m[0] + Wi[0][1] * m[1], Wi[1][0] * m[0] + Wi[1][1] * m[1]]
        want = c[0] * 2 + c[1] * 3 - 0.5 * (c[0] * m[0] + c[1] * m[1]) + math.log(0.5)
        assert r.scores[0][k] == pytest.approx(want, abs=1e-9)
    assert "apparent_error" not in r


def test_outliers_do_not_move_centres():
    r = robust_lda(X, y)
    assert r.centers[0][0] < 1 and r.centers[1][0] > 5
    with pytest.raises(ValueError):
        robust_lda(X, [0] * 14)
