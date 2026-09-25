"""Tests for strtwt.stratified_weights (Cole & Hernan 2008)."""

import math
import statistics

import pytest

from morie.fn.strtwt import stratified_weights


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _data(n=40):
    S = [[float(k % 2)] for k in range(n)]
    H = [[math.sin(1.7 * k)] for k in range(n)]
    A = [1.0 if ((29 * k + 3) % 67 + 0.5) / 67 < _expit(0.3 + 0.8 * h[0] - 0.5 * s[0]) else 0.0
         for k, (h, s) in enumerate(zip(H, S))]
    return A, H, S


def _logit_fit(Z, a):
    """Logistic MLE by Newton-Raphson to convergence (3 columns)."""
    p = len(Z[0])
    b = [0.0] * p
    for _ in range(100):
        mu = [_expit(sum(x * c for x, c in zip(z, b))) for z in Z]
        g = [sum(z[r] * (y - m) for z, y, m in zip(Z, a, mu)) for r in range(p)]
        I = [[sum(z[r] * z[c] * m * (1 - m) for z, m in zip(Z, mu)) for c in range(p)] for r in range(p)]
        M = [row[:] + [g[i]] for i, row in enumerate(I)]
        for c in range(p):
            for r in range(p):
                if r != c:
                    f = M[r][c] / M[c][c]
                    M[r] = [u - f * v for u, v in zip(M[r], M[c])]
        b = [bi + M[i][p] / M[i][i] for i, bi in enumerate(b)]
    return [_expit(sum(x * c for x, c in zip(z, b))) for z in Z]


def test_strtwt_basic():
    """Numerator f(A|S): with one binary stratum variable the logistic
    fit is saturated, so P(A=1|S=s) is the stratum's treated fraction;
    denominator f(A|H,S) by an independent Newton fit; sw = num/den."""
    A, H, S = _data()
    frac = {s: statistics.fmean(a for a, t in zip(A, S) if t[0] == s) for s in (0.0, 1.0)}
    num = [frac[s[0]] if a else 1 - frac[s[0]] for a, s in zip(A, S)]
    pd = _logit_fit([[1.0, s[0], h[0]] for s, h in zip(S, H)], A)
    den = [p if a else 1 - p for a, p in zip(A, pd)]
    r = stratified_weights(A, H, S)
    assert r["num"] == pytest.approx(num, abs=1e-12)
    assert r["den"] == pytest.approx(den, abs=1e-10)
    sw = [x / y for x, y in zip(num, den)]
    assert r["weights"] == pytest.approx(sw, rel=1e-9)
    assert r["unstabilized"] == pytest.approx([1 / d for d in den], rel=1e-9)
    assert r["estimate"] == pytest.approx(statistics.fmean(sw), rel=1e-9)
    assert r["sd"] == pytest.approx(statistics.stdev(sw), rel=1e-8)


def test_strtwt_edge():
    """No history: numerator and denominator coincide and every weight is
    one (the canonical case); non-binary A and ragged covariates raise."""
    A = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    S = [[0.0], [0.0], [1.0], [1.0], [0.0], [1.0], [1.0], [0.0]]
    r = stratified_weights(A, None, S)
    assert r["weights"] == pytest.approx([1.0] * 8, abs=1e-12)
    with pytest.raises(ValueError):
        stratified_weights([0.5, 1.0], None, None)
    with pytest.raises(ValueError):
        stratified_weights(A, [[1.0]] * 3, S)
