"""Tests for survlnk.link_function_survival (discrete-time hazard GLM)."""

import math

import pytest

from morie.fn.survlnk import link_function_survival


def _data(n=25):
    x = [math.sin(1.1 * k) for k in range(n)]
    t = [float((k * 5) % 9 + 1) for k in range(n)]
    e = [1.0 if (k % 4 != 0 and t[k] < 9) else 0.0 for k in range(n)]
    return t, e, x


def _pp(t, e, x):
    """Person-period expansion: one Bernoulli row per subject per event
    time at which it is still at risk."""
    et = sorted({ti for ti, ei in zip(t, e) if ei})
    rows = [(k, xi, 1.0 if (ti == tk and ei) else 0.0)
            for ti, ei, xi in zip(t, e, x) for k, tk in enumerate(et) if ti >= tk]
    return et, rows


def _h_dh(eta, link):
    if link == "cloglog":
        h = 1 - math.exp(-math.exp(eta))
        return h, math.exp(eta) * (1 - h)
    h = 1 / (1 + math.exp(-eta))
    return h, h * (1 - h)


@pytest.mark.parametrize("link", ["cloglog", "logit"])
def test_survlnk_basic(link):
    """At the returned (alpha, beta) the binomial score of the
    person-period data vanishes, and se(beta) is the beta entry of the
    inverse expected information sum dh^2/(h(1-h)) g g'.  R's
    glm(y ~ factor(k) + x - 1, binomial(link)) on the same expansion
    gives beta -0.039454950283 (cloglog) and -0.041089571119 (logit)."""
    t, e, x = _data()
    r = link_function_survival(t, e, x, link)
    et, rows = _pp(t, e, x)
    K = len(et)
    th = [float(v) for v in r["alpha"]] + [float(r["estimate"][0])]
    U = [0.0] * (K + 1)
    I = [[0.0] * (K + 1) for _ in range(K + 1)]
    for k, xi, y in rows:
        h, dh = _h_dh(th[k] + th[K] * xi, link)
        g = [0.0] * (K + 1)
        g[k], g[K] = 1.0, xi
        for a in range(K + 1):
            U[a] += (y - h) / (h * (1 - h)) * dh * g[a]
            for b in range(K + 1):
                I[a][b] += dh * dh / (h * (1 - h)) * g[a] * g[b]
    assert max(abs(u) for u in U) < 1e-8
    M = [row[:] + [1.0 if i == K else 0.0] for i, row in enumerate(I)]
    for c in range(K + 1):
        p = max(range(c, K + 1), key=lambda q: abs(M[q][c]))
        M[c], M[p] = M[p], M[c]
        for q in range(K + 1):
            if q != c:
                f = M[q][c] / M[c][c]
                M[q] = [u - f * v for u, v in zip(M[q], M[c])]
    vK = M[K][K + 1] / M[K][K]
    assert float(r["se"][0]) == pytest.approx(math.sqrt(vK), rel=1e-8)
    assert [float(v) for v in r["event_times"]] == et
    if link == "logit":
        assert float(r["estimate"][0]) == pytest.approx(-0.041089571119, abs=1e-9)


def test_survlnk_edge():
    """No events and an unknown link raise."""
    t, e, x = _data()
    with pytest.raises(ValueError):
        link_function_survival(t, [0.0] * len(t), x)
    with pytest.raises(ValueError):
        link_function_survival(t, e, x, "probit")
