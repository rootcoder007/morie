"""Tests for theteap.theta_eap (EAP ability, Bock & Mislevy 1982)."""

import math

import pytest


ITEMS = [[1.2, -0.8, 0.0], [0.9, -0.2, 0.1], [1.5, 0.3, 0.2], [0.7, 1.0, 0.0], [1.1, 1.6, 0.15]]
X = [[1, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]]


def _logpost(t, y, mu=0.0, sd=1.0):
    lp = -(t - mu) ** 2 / (2 * sd * sd)
    for (a, b, c), r in zip(ITEMS, y):
        p = c + (1 - c) / (1 + math.exp(-a * (t - b)))
        lp += math.log(p) if r else math.log(1 - p)
    return lp

from morie.fn.theteap import theta_eap


def _eap(y):
    """Posterior mean and SD by composite Simpson on [-9, 9] with 18000
    panels: the integrand is smooth and the N(0,1) prior leaves < 1e-17
    of mass outside, so this is exact to ~1e-12."""
    m, h = 18000, 18.0 / 18000
    ts = [-9 + i * h for i in range(m + 1)]
    lp = [_logpost(t, y) for t in ts]
    top = max(lp)
    w = [(1 if i in (0, m) else (4 if i % 2 else 2)) * math.exp(v - top) for i, v in enumerate(lp)]
    z = sum(w)
    mean = sum(wi * t for wi, t in zip(w, ts)) / z
    var = sum(wi * (t - mean) ** 2 for wi, t in zip(w, ts)) / z
    return mean, math.sqrt(var)


def test_theteap_basic():
    """EAP and posterior SD per examinee match independent Simpson
    integration.  61-node Gauss-Hermite on these smooth posteriors is
    accurate to below 1e-7, the tolerance used."""
    r = theta_eap(X, ITEMS)
    for i, y in enumerate(X):
        m, s = _eap(y)
        assert float(r["theta"][i]) == pytest.approx(m, abs=1e-7)
        assert float(r["se"][i]) == pytest.approx(s, abs=1e-7)
    assert r["n_examinees"] == 4 and r["n_items"] == 5


def test_theteap_edge():
    """One examinee returns scalars; the all-correct pattern sits above
    the all-wrong one; a mismatched item table raises."""
    r = theta_eap([X[1]], ITEMS)
    assert isinstance(r["theta"], float)
    assert r["theta"] > theta_eap([X[2]], ITEMS)["theta"]
    with pytest.raises(ValueError):
        theta_eap(X, ITEMS[:4])
