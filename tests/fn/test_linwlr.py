"""Tests for linwlr.

Replaces a generated test that called a stub returning mean(y). The
full anchor is ledger/wave3/anchor_msm8.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as kc
from morie.fn.linwlr import linear_weighted_learner

N = 3000


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))



@pytest.fixture(scope="module")
def tv():
    """Time-varying treatment with confounder feedback."""
    rng = np.random.default_rng(20260813)
    L0 = [rng.standard_normal() for _ in range(N)]
    A0 = [1.0 if rng.uniform() < expit(-0.3 + 1.2 * L0[i]) else 0.0
          for i in range(N)]
    L1 = [1.0 * A0[i] + 0.6 * L0[i] + rng.standard_normal()
          for i in range(N)]
    A1 = [1.0 if rng.uniform() < expit(-0.2 + 1.0 * L1[i] + 0.8 * A0[i])
          else 0.0 for i in range(N)]
    Y = [1.0 + 0.6 * A0[i] + 1.5 * A1[i] + 0.9 * L1[i] + 0.7 * L0[i]
         + 0.7 * rng.standard_normal() for i in range(N)]
    V = [1.0 if rng.uniform() < 0.5 else 0.0 for _ in range(N)]
    Yv = [Y[i] + 1.0 * V[i] * (A0[i] + A1[i]) + 0.4 * V[i]
          for i in range(N)]
    Ys = [1.0 + 2.0 * A0[i] + 3.0 * L0[i] + 0.5 * rng.standard_normal()
          for i in range(N)]
    Yh = [1.0 + (2.0 + 1.5 * L0[i]) * A0[i] + 3.0 * L0[i]
          + 0.5 * rng.standard_normal() for i in range(N)]
    return {"A": [A0, A1], "L": [[[v] for v in L0], [[v] for v in L1]],
            "Ls": [[v] for v in L0], "A0": A0,
            "Y": Y, "V": V, "Yv": Yv, "Ys": Ys, "Yh": Yh}


def test_constant_blip_recovered_by_both_routes(tv):
    g = linear_weighted_learner(tv["Ys"], tv["A0"], None,
                                pi_covariates=tv["Ls"])
    w = linear_weighted_learner(tv["Ys"], tv["A0"], None,
                                pi_covariates=tv["Ls"], method="wls")
    assert g["estimate"] == pytest.approx(2.0, abs=0.12)
    assert w["estimate"] == pytest.approx(2.0, abs=0.15)
    # the crude estimate must be badly wrong, or the check is vacuous
    crude = kc.wls([[a] for a in tv["A0"]], tv["Ys"], [1.0] * N)
    assert abs(crude["coef"][1] - 2.0) > 0.3


def test_blip_varying_with_w(tv):
    g = linear_weighted_learner(tv["Yh"], tv["A0"], tv["Ls"],
                                pi_covariates=tv["Ls"])
    assert g["psi"][0] == pytest.approx(2.0, abs=0.15)
    assert g["psi"][1] == pytest.approx(1.5, abs=0.15)


def test_argument_checks(tv):
    with pytest.raises(ValueError):
        linear_weighted_learner(tv["Ys"], tv["A0"], None,
                                propensity=[0.0] * N)
    with pytest.raises(ValueError):
        linear_weighted_learner(tv["Ys"], tv["A0"], None, method="nope")
    with pytest.raises(ValueError):
        linear_weighted_learner(tv["Ys"], tv["A0"][:-1], None)
