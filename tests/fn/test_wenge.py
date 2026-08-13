"""Tests for wenge.

Replaces a generated test that called a stub returning mean(y). The
full anchor is ledger/wave3/anchor_msm8.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wenge import mediation_functional, weight_based_mediation

N = 3000


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))



@pytest.fixture(scope="module")
def mediation():
    """All-discrete, so every conditional can be fitted saturated and
    Tchetgen Tchetgen & Shpitser's equivalence must hold exactly."""
    rng = np.random.default_rng(20260813)
    C = [[1.0 if rng.uniform() < 0.5 else 0.0] for _ in range(N)]
    E = [1.0 if rng.uniform() < 0.3 + 0.4 * C[i][0] else 0.0
         for i in range(N)]
    M = [[1.0 if rng.uniform() < 0.2 + 0.3 * E[i] + 0.3 * C[i][0]
          else 0.0] for i in range(N)]
    Y = [1.0 + 2.0 * E[i] + 1.5 * M[i][0] + 0.8 * C[i][0]
         + rng.standard_normal() for i in range(N)]
    Ynm = [1.0 + 2.0 * E[i] + 0.8 * C[i][0] + rng.standard_normal()
           for i in range(N)]
    # E raises P(M=1) by 0.3 and M raises Y by 1.5, so NIE = 0.45
    return {"C": C, "E": E, "M": M, "Y": Y, "Ynm": Ynm,
            "nde": 2.0, "nie": 0.45}


def test_three_strategies_coincide_on_a_saturated_model(mediation):
    """Tchetgen Tchetgen & Shpitser: theta_em = theta_ye = theta_ym on
    the nonparametric model. They are the same number, not three
    estimators that happen to be close."""
    t = mediation_functional(mediation["Y"], mediation["E"],
                             mediation["M"], mediation["C"],
                             strategy="all", saturated=True)
    assert t["em"] == pytest.approx(t["ye"], abs=1e-9)
    assert t["em"] == pytest.approx(t["ym"], abs=1e-9)


def test_natural_effects_recovered(mediation):
    r = weight_based_mediation(mediation["E"], mediation["M"],
                               mediation["C"], mediation["Y"])
    assert r["nde"] == pytest.approx(mediation["nde"], abs=0.12)
    assert r["nie"] == pytest.approx(mediation["nie"], abs=0.12)
    # the decomposition must be exact, not approximate
    assert r["nde"] + r["nie"] == pytest.approx(r["total"], abs=1e-9)


def test_no_mediation_gives_zero_indirect_effect(mediation):
    r = weight_based_mediation(mediation["E"], mediation["M"],
                               mediation["C"], mediation["Ynm"])
    assert abs(r["nie"]) < 0.1


def test_argument_checks(mediation):
    with pytest.raises(ValueError):
        mediation_functional(mediation["Y"], [2.0] * N, mediation["M"],
                             mediation["C"])
    with pytest.raises(ValueError):
        mediation_functional(mediation["Y"], mediation["E"],
                             mediation["M"], mediation["C"],
                             strategy="nope")
