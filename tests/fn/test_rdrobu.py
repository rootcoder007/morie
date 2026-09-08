"""Tests for rdrobu (CCT 2014 interval front end)."""

import math

from morie.fn.causrddc import causrddc
from morie.fn.rdrobu import (calonico_cattaneo_titiunik,
                             rd_confidence_intervals, rdrobu)


def _make(n=800, seed=7, tau=1.0, noise=0.3):
    state = [seed]

    def rnd():
        state[0] = (1103515245 * state[0] + 12345) % (1 << 31)
        return state[0] / float(1 << 31)

    x, y = [], []
    for _ in range(n):
        xi = 2.0 * rnd() - 1.0
        u1 = max(rnd(), 1e-12)
        e = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * rnd())
        x.append(xi)
        y.append(0.5 * xi + 0.8 * xi * xi + (tau if xi >= 0 else 0.0) +
                 noise * e)
    return x, y


X, Y = _make()


def test_front_end_matches_the_shared_implementation():
    for kw in ({}, dict(p=2), dict(h=0.4, b=0.6), dict(kernel="uniform")):
        a = rdrobu(Y, X, **kw)
        b = causrddc(Y, X, **kw)
        assert abs(a["estimate"] - b["estimate"]) < 1e-12
        assert abs(a["bias_corrected"] - b["bias_corrected"]) < 1e-12
        assert a["intervals"]["robust"] == b["ci_robust"]
        assert a["intervals"]["conventional"] == b["ci_conventional"]


def test_interval_relations():
    r = rdrobu(Y, X)
    c = r["intervals"]["conventional"]
    b = r["intervals"]["bias_corrected"]
    rb = r["intervals"]["robust"]
    assert abs((c[1] - c[0]) - (b[1] - b[0])) < 1e-12
    assert rb[1] - rb[0] > b[1] - b[0]
    assert abs((rb[0] + rb[1]) / 2 - (b[0] + b[1]) / 2) < 1e-12
    assert abs(r["correction_factor"] -
               r["se_robust"] / r["se_conventional"]) < 1e-12
    assert r["correction_factor"] > 1.0
    assert abs(r["bias_estimate"] -
               (r["estimate"] - r["bias_corrected"])) < 1e-12


def test_remark_7_through_the_front_end():
    lo = rdrobu(Y, X, p=1, h=0.5, b=0.5)
    hi = rdrobu(Y, X, p=2, h=0.5)
    assert abs(lo["bias_corrected"] - hi["estimate"]) < 1e-11


def test_validation_is_inherited():
    for call in (lambda: rdrobu(Y, X, kernel="gaussian"),
                 lambda: rdrobu(Y, X, alpha=0.0),
                 lambda: rdrobu(Y[:-1], X)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_aliases():
    assert rd_confidence_intervals is rdrobu
    assert abs(calonico_cattaneo_titiunik(Y, X, 0.0)["estimate"] -
               rdrobu(Y, X)["estimate"]) < 1e-12
