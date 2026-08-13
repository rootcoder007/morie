"""Tests for mfovsm.

Replaces a generated test that called a stub returning mean(y). The
full anchor is ledger/wave3/anchor_msm8.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.mfovsm import mfo_vsm

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


def test_effect_modification_recovered(tv):
    m = mfo_vsm(tv["Yv"], tv["V"], tv["A"], tv["L"])
    assert m["estimate"] == pytest.approx(1.0, abs=0.2)


def test_v_in_the_numerator_changes_weights_not_the_estimand(tv):
    """Hernan & Robins Sec. 12.5: putting V in the weight numerator is
    a variance move, not an estimand move."""
    a = mfo_vsm(tv["Yv"], tv["V"], tv["A"], tv["L"])
    b = mfo_vsm(tv["Yv"], tv["V"], tv["A"], tv["L"],
                v_in_numerator=False)
    assert a["estimate"] == pytest.approx(b["estimate"], abs=0.05)
    assert max(abs(a["weights"][i] - b["weights"][i])
               for i in range(N)) > 1e-6


def test_no_modification_gives_beta3_near_zero(tv):
    m = mfo_vsm(tv["Y"], tv["V"], tv["A"], tv["L"])
    assert abs(m["estimate"]) < 3.0 * m["se"]


def test_argument_checks(tv):
    with pytest.raises(ValueError):
        mfo_vsm(tv["Yv"], tv["V"][:-1], tv["A"], tv["L"])
    with pytest.raises(ValueError):
        mfo_vsm(tv["Yv"], tv["V"], tv["A"], tv["L"], contrast="nope")
