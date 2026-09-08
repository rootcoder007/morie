"""Tests for shdsmw.

Replaces a generated test that called a stub returning mean(y). The
full anchor is ledger/wave3/anchor_msm8.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.shdsmw import penalty_path, shrinkage_msm

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


def test_penalty_zero_is_the_unpenalized_msm(tv):
    r = shrinkage_msm(tv["Y"], tv["A"], tv["L"], lam=0.0)
    row = [p for p in r["path"] if p["lam"] == 0.0][0]
    assert row["estimate"] == pytest.approx(r["estimate"], abs=1e-12)


def test_the_penalty_actually_bites(tv):
    """A flat path means the penalty never reached the propensity
    model, which is the bug this check was written to catch."""
    r = shrinkage_msm(tv["Y"], tv["A"], tv["L"], lam=0.0)
    p = {row["lam"]: row for row in r["path"]}
    assert abs(p[1000.0]["estimate"] - p[0.0]["estimate"]) > 0.1
    # and it moves TOWARD the unadjusted estimate, not anywhere
    assert (abs(p[1000.0]["estimate"] - r["unadjusted"])
            < abs(p[0.0]["estimate"] - r["unadjusted"]))
    # Setoguchi / Westreich's trade: bias up, variance down
    assert (p[1000.0]["effective_sample_size"]
            > 1.2 * p[0.0]["effective_sample_size"])


def test_argument_checks(tv):
    with pytest.raises(ValueError):
        shrinkage_msm(tv["Y"], tv["A"], tv["L"], lam=-1.0)
    with pytest.raises(ValueError):
        shrinkage_msm(tv["Y"], tv["A"], tv["L"], contrast="nope")
    assert len(penalty_path(tv["Y"], tv["A"], tv["L"])) > 1
