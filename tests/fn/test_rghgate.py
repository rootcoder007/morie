"""Tests for rghgate.rangayyan_hh_gating."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_hh_gating


def _rates(v):
    am = 0.1 * (v + 40) / (1 - math.exp(-(v + 40) / 10))
    bm = 4 * math.exp(-(v + 65) / 18)
    ah = 0.07 * math.exp(-(v + 65) / 20)
    bh = 1 / (1 + math.exp(-(v + 35) / 10))
    an = 0.01 * (v + 55) / (1 - math.exp(-(v + 55) / 10))
    bn = 0.125 * math.exp(-(v + 65) / 80)
    return am, bm, ah, bh, an, bn


def test_rghgate_basic():
    """At rest (-65 mV) the gates sit at alpha / (alpha + beta): the
    textbook m 0.0529, h 0.5961, n 0.3177."""
    r = rangayyan_hh_gating(-65.0)
    am, bm, ah, bh, an, bn = _rates(-65.0)
    assert r["m"] == pytest.approx(am / (am + bm), rel=1e-14)
    assert r["h"] == pytest.approx(ah / (ah + bh), rel=1e-14)
    assert r["n"] == pytest.approx(an / (an + bn), rel=1e-14)
    assert (round(r["m"], 4), round(r["h"], 4), round(r["n"], 4)) == (0.0529, 0.5961, 0.3177)
    assert r["tau_m_ms"] < r["tau_n_ms"] and r["tau_m_ms"] < r["tau_h_ms"]


def test_rghgate_edge():
    """A clamped step is the exact exponential relaxation, and the rates
    are continuous through their removable singularities."""
    r = rangayyan_hh_gating(-20.0, dt=0.05, m=0.05, h=0.6, n=0.32, steps=7)
    am, bm, ah, bh, an, bn = _rates(-20.0)
    for g, a, b, x0 in (("m", am, bm, 0.05), ("h", ah, bh, 0.6), ("n", an, bn, 0.32)):
        xinf, tau = a / (a + b), 1 / (a + b)
        assert r[g] == pytest.approx(xinf + (x0 - xinf) * math.exp(-0.35 / tau), rel=1e-12)
    for v0 in (-40.0, -55.0):
        at = rangayyan_hh_gating(v0)
        near = rangayyan_hh_gating(v0 + 1e-5)
        assert at["m"] == pytest.approx(near["m"], rel=1e-4)
        assert at["n"] == pytest.approx(near["n"], rel=1e-4)
    with pytest.raises(ValueError, match="dt must be positive"):
        rangayyan_hh_gating(-65.0, dt=0.0)


