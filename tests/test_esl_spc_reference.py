"""Supervised principal components (ESL Alg. 18.1) against prcomp + lm in R."""

import math

from morie.fn.eslsup import esl_supervised_pc


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_supervised_pc():
    i = range(1, 41)
    X = [
        [
            math.sin(t),
            math.cos(t),
            math.sin(2 * t),
            math.cos(3 * t),
            0.5 * math.sin(t) + 0.5 * math.cos(t),
            math.sin(5 * t),
        ]
        for t in i
    ]
    y = [r[0] + r[1] + 0.3 * math.cos(7 * t) for r, t in zip(X, i)]
    r = esl_supervised_pc(X, y, 1.5, newdata=[[0.2, 0.5, 0.1, -0.3, 0.35, 0.2]])
    ref_s = (
        4.558648683265714929,
        4.276489864696869425,
        0.280854819590449944,
        -0.306350571300062324,
        6.258940335774668462,
        -0.039430204696697398,
    )
    assert all(close(a, b) for a, b in zip(r["scores_univariate"], ref_s)) and r["selected"] == [0, 1, 4]
    assert all(
        close(a, b) for a, b in zip(r["fitted"][:3], (1.36828056407229570, 0.52826659213813765, -0.80912088572445917))
    )
    assert close(r["prediction"][0], 0.66901672530837319)
    Xf = [[r[0], r[1], r[2], r[3], r[5]] for r in X]  # drop the collinear fifth column
    full = esl_supervised_pc(Xf, y, 0.0, m=5)  # all features and all components: ordinary least squares
    from morie.fn.linsys import _householder_ls

    b, _ = _householder_ls([[1.0] + row for row in Xf], y)
    assert all(close(f, sum(bb * v for bb, v in zip(b, [1.0] + row)), 1e-10) for f, row in zip(full["fitted"], Xf))
