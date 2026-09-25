"""Tests for morebs.empirical_bayes_moran (Assuncao & Reis 1999 EBI)."""

import math

import pytest

from morie.fn.morebs import empirical_bayes_moran

O = [3, 10, 1, 7, 0, 5]
N = [120, 300, 80, 150, 60, 200]
W = [[0.0] * 6 for _ in range(6)]
for _i in range(5):
    W[_i][_i + 1] = W[_i + 1][_i] = 1.0
W[0][5] = W[5][0] = 1.0


def test_morebs_basic():
    """EB deviates z_i = (p_i - b) / sqrt(a + b / n_i) with Marshall's
    moment estimators, then Moran I on the centred z, recomputed.
    spdep::EBImoran.mc on the same ring gives -0.68815535981680098."""
    r = empirical_bayes_moran(O, N, W)
    assert isinstance(r, dict)
    m, sn = 6, sum(N)
    p = [o / n for o, n in zip(O, N)]
    b = sum(O) / sn
    s2 = sum(n * (q - b) ** 2 for n, q in zip(N, p)) / sn
    a = max(0.0, s2 - b / (sn / m))
    z = [(q - b) / math.sqrt(a + b / n) for q, n in zip(p, N)]
    zb = sum(z) / m
    zt = [v - zb for v in z]
    s0 = sum(map(sum, W))
    ebi = m / s0 * sum(zt[i] * sum(W[i][j] * zt[j] for j in range(m)) for i in range(m)) \
        / sum(v * v for v in zt)
    assert r["statistic"] == pytest.approx(ebi, rel=1e-13)
    assert r["statistic"] == pytest.approx(-0.68815535981680098, rel=1e-13)


def test_morebs_edge():
    """Mismatched lengths and a zero population are refused."""
    with pytest.raises(ValueError):
        empirical_bayes_moran(O[:5], N, W)
    with pytest.raises(ValueError):
        empirical_bayes_moran(O, N[:5] + [0], W)
