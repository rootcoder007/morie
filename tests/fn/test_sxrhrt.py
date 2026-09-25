"""Tests for sxrhrt.sex_specific_h2 (bivariate REML across sexes)."""

import math

import pytest

from morie.fn.sxrhrt import sex_specific_h2


def _data(n=16, m=6):
    """A GRM from standardised pseudo-random markers, and a phenotype with
    a sex mean difference."""
    G = [[float(int(((math.sin(12.9898 * (i * m + j) + 7.1) * 43758.5453) % 1) * 3)) for j in range(m)]
         for i in range(n)]
    cols = list(zip(*G))
    Z = []
    for c in cols:
        mu = sum(c) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in c) / n) or 1.0
        Z.append([(v - mu) / sd for v in c])
    K = [[sum(Z[a][i] * Z[a][j] for a in range(m)) / m for j in range(n)] for i in range(n)]
    sex = [1 if i % 2 else 2 for i in range(n)]
    g = [sum(0.4 * Z[a][i] for a in range(0, m, 2)) for i in range(n)]
    y = [(1.5 if s == 1 else 0.0) + gi + 0.9 * math.sin(3.7 * i) for i, (s, gi) in enumerate(zip(sex, g))]
    return y, sex, K


def _reml(th, y, K, male):
    """-1/2 (log|V| + log|X'V^-1 X| + r'V^-1 r), X = sex-specific
    intercepts, by dense Gaussian elimination."""
    s2gm, s2gf, rg, s2em, s2ef = th
    n = len(y)
    c = rg * math.sqrt(s2gm * s2gf)
    V = [[(s2gm if male[i] and male[j] else s2gf if not (male[i] or male[j]) else c) * K[i][j]
          + ((s2em if male[i] else s2ef) if i == j else 0.0) for j in range(n)] for i in range(n)]
    X = [[1.0, 0.0] if male[i] else [0.0, 1.0] for i in range(n)]
    M = [V[i][:] + [X[i][0], X[i][1], y[i]] for i in range(n)]
    ld = 0.0
    for col in range(n):
        piv = M[col][col]
        ld += math.log(piv)
        for r in range(col + 1, n):
            f = M[r][col] / piv
            M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    sol = [[0.0] * 3 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for k in range(3):
            sol[i][k] = (M[i][n + k] - sum(M[i][j] * sol[j][k] for j in range(i + 1, n))) / M[i][i]
    XtViX = [[sum(X[i][a] * sol[i][b] for i in range(n)) for b in range(2)] for a in range(2)]
    XtViy = [sum(X[i][a] * sol[i][2] for i in range(n)) for a in range(2)]
    det = XtViX[0][0] * XtViX[1][1] - XtViX[0][1] * XtViX[1][0]
    beta = [(XtViX[1][1] * XtViy[0] - XtViX[0][1] * XtViy[1]) / det,
            (XtViX[0][0] * XtViy[1] - XtViX[1][0] * XtViy[0]) / det]
    r = [y[i] - X[i][0] * beta[0] - X[i][1] * beta[1] for i in range(n)]
    Vir = [sol[i][2] - sol[i][0] * beta[0] - sol[i][1] * beta[1] for i in range(n)]
    return -0.5 * (ld + math.log(det) + sum(a * b for a, b in zip(r, Vir))), beta


def test_sxrhrt_basic():
    """The reported restricted log-likelihood and sex-specific
    intercepts are recomputed independently at the returned parameters,
    no single-coordinate move of 1e-3 (log scale for variances) raises
    it, and h2 = s2g/(s2g + s2e) in each sex."""
    y, sex, K = _data()
    male = [s == 1 for s in sex]
    r = sex_specific_h2(y, sex, K)
    th = [r["sigma2_g_male"], r["sigma2_g_female"], r["rg"], r["sigma2_e_male"], r["sigma2_e_female"]]
    ll, beta = _reml(th, y, K, male)
    assert r["reml_loglik"] == pytest.approx(ll, abs=1e-8)
    assert [float(b) for b in r["coefficients"]] == pytest.approx(beta, abs=1e-8)
    for i in (0, 1, 3, 4):
        for sgn in (1, -1):
            t = list(th)
            t[i] *= math.exp(sgn * 1e-3)
            assert _reml(t, y, K, male)[0] <= ll + 1e-9
    for sgn in (1, -1):
        t = list(th)
        t[2] = max(min(t[2] + sgn * 1e-3, 0.999), -0.999)
        assert _reml(t, y, K, male)[0] <= ll + 1e-9
    assert r["h2_male"] == pytest.approx(th[0] / (th[0] + th[3]), rel=1e-15)
    assert r["h2_female"] == pytest.approx(th[1] / (th[1] + th[4]), rel=1e-15)
    assert r["lrt_equal_h2"] >= 0.0 and r["lrt_rg_equals_one"] >= 0.0


def test_sxrhrt_edge():
    """One sex with fewer than two members, a mis-sized or asymmetric K
    raise."""
    y, sex, K = _data()
    with pytest.raises(ValueError):
        sex_specific_h2(y, [1] + [2] * (len(y) - 1), K)
    with pytest.raises(ValueError):
        sex_specific_h2(y, sex, [row[:-1] for row in K])
    Ka = [row[:] for row in K]
    Ka[0][1] += 0.1
    with pytest.raises(ValueError):
        sex_specific_h2(y, sex, Ka)


