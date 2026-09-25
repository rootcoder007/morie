"""Tests for bsaphys.rangayyan_goldman_eqn (Goldman-Hodgkin-Katz)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_goldman_eqn


C = {"K_out": 4.0, "K_in": 140.0, "Na_out": 145.0, "Na_in": 12.0, "Cl_out": 116.0, "Cl_in": 4.0}


def test_rgghk_basic():
    """V = (RT/F) ln[(PK Ko + PNa Nao + PCl Cli) / (PK Ki + PNa Nai + PCl Clo)]
    with CODATA R and F; chloride enters with its concentrations swapped."""
    R, F, T = 8.314462618, 96485.33212, 310.15
    num = 1.0 * 4 + 0.04 * 145 + 0.45 * 4
    den = 1.0 * 140 + 0.04 * 12 + 0.45 * 116
    v = R * T / F * math.log(num / den)
    r = rangayyan_goldman_eqn(C)
    assert r["numerator_mM"] == pytest.approx(num, rel=1e-15)
    assert r["denominator_mM"] == pytest.approx(den, rel=1e-15)
    assert r["potential_V"] == pytest.approx(v, rel=1e-9)
    assert r["potential_mV"] == pytest.approx(1000 * v, rel=1e-9)


def test_rgghk_edge():
    """Only potassium permeable: the K Nernst potential; a missing ion
    raises."""
    R, F, T = 8.314462618, 96485.33212, 310.15
    r = rangayyan_goldman_eqn(C, P_K=1.0, P_Na=0.0, P_Cl=0.0)
    assert r["potential_V"] == pytest.approx(R * T / F * math.log(4 / 140), rel=1e-9)
    with pytest.raises(ValueError):
        rangayyan_goldman_eqn({"K_out": 4.0})
