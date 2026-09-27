"""Generalised moments spatial error model and GS2SLS SAC (Kelejian and Prucha).

Checked against spatialreg 1.3 GMerrorsar and gstsls (non-legacy) to 1e-10
(the limit of spatialreg's nlminb stop); the cross test in
tests/cross/test-morie_vs_spatialreg.R repeats that in R.
"""

import pytest

from morie.fn.gmsar import gm_error_sar, gs2sls_sac, kp_moments

N = 8
Y = [1.0, 2.2, 1.4, 3.1, 0.9, 2.0, 2.6, 1.1]
XV = [0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4]


def _w():
    W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
    return [[v / sum(r) for v in r] for r in W]


def _mv(W, v):
    return [sum(a * b for a, b in zip(r, v)) for r in W]


def _solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _ols(X, y):
    p = len(X[0])
    return _solve(
        [[sum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)],
        [sum(r[a] * t for r, t in zip(X, y)) for a in range(p)],
    )


def test_kp_moments_by_hand():
    u = [0.3, -0.2, 0.5, -0.1, 0.4, -0.6, 0.2, -0.5]
    W = _w()
    G, g, trwpw, wu, wwu = kp_moments(u, W)
    assert wu == pytest.approx(_mv(W, u), abs=1e-15)
    assert g[0] == pytest.approx(sum(v * v for v in u) / N, abs=1e-15)
    assert g[2] == pytest.approx(sum(a * b for a, b in zip(u, wu)) / N, abs=1e-15)
    assert trwpw == pytest.approx(sum(v * v for r in W for v in r), abs=1e-15)
    assert [r[2] for r in G] == pytest.approx([1.0, trwpw / N, 0.0], abs=1e-15)


def test_gm_error_sar_solves_the_moments_and_fgls():
    W = _w()
    X = [[1.0, v] for v in XV]
    r = gm_error_sar(Y, X, W).value
    lam = r["lambda"]
    assert round(lam, 6) == -0.042117
    b0 = _ols(X, Y)
    e = [Y[i] - X[i][0] * b0[0] - X[i][1] * b0[1] for i in range(N)]
    G, g, *_ = kp_moments(e, W)
    # derivative of the profiled criterion is zero at lambda
    c3 = [row[2] for row in G]
    rr = [g[i] - G[i][0] * lam - G[i][1] * lam**2 for i in range(3)]
    d1 = [-G[i][0] - 2 * G[i][1] * lam for i in range(3)]
    c33 = sum(v * v for v in c3)
    cr = sum(a * b for a, b in zip(c3, rr))
    f1 = 2 * sum(a * b for a, b in zip(rr, d1)) - 2 * cr * sum(a * b for a, b in zip(c3, d1)) / c33
    assert abs(f1) < 1e-12
    assert r["sigma2_gm"] == pytest.approx(cr / c33, abs=1e-12)
    Wy = _mv(W, Y)
    WX = [_mv(W, [row[k] for row in X]) for k in range(2)]
    B = [[X[i][k] - lam * WX[k][i] for k in range(2)] for i in range(N)]
    beta = _ols(B, [Y[i] - lam * Wy[i] for i in range(N)])
    assert r["coefficients"] == pytest.approx(beta, abs=1e-12)
    assert r["lambda_se"] > 0
    sr = gm_error_sar(Y, X, W, lambda_se_method="spatialreg").value
    assert sr["coefficients"] == r["coefficients"]
    assert sr["lambda_se"] != r["lambda_se"]
    with pytest.raises(ValueError):
        gm_error_sar(Y, X, W, lambda_se_method="other")


def test_gs2sls_sac_final_stage_is_2sls_on_transformed_data():
    W = _w()
    X = [[1.0, v] for v in XV]
    r = gs2sls_sac(Y, X, W).value
    assert round(r["rho"], 6) == -0.561259
    lam = r["lambda"]
    Wy = _mv(W, Y)
    wx = _mv(W, XV)
    inst = [[wx[i], v] for i, v in enumerate(_mv(W, wx))]
    t = lambda v: [a - lam * b for a, b in zip(v, _mv(W, v))]  # noqa: E731
    yt, wyt, c0, c1 = t(Y), t(Wy), t([1.0] * N), t(XV)
    Q = [[c0[i], c1[i]] + inst[i] for i in range(N)]
    Qc = [list(c) for c in zip(*Q)]
    bz = _solve(
        [[sum(a * b for a, b in zip(u, v)) for v in Qc] for u in Qc], [sum(a * b for a, b in zip(u, wyt)) for u in Qc]
    )
    yhat = [sum(a * b for a, b in zip(row, bz)) for row in Q]
    Zp = [[yhat[i], c0[i], c1[i]] for i in range(N)]
    assert r["coefficients"] == pytest.approx(_ols(Zp, yt), abs=1e-10)


def test_gs2sls_sac_needs_a_regressor():
    with pytest.raises(ValueError):
        gs2sls_sac(Y, [[1.0] for _ in range(N)], _w())
