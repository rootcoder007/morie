"""Tests for tmlcps -- Kennedy, Ma, McHugh & Small (2017).

Replaces a generated test that called a stub returning mean(y). The
full anchor, including both double-robustness arms, is
ledger/wave3/anchor_tmlcps.py.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as kc
from morie.fn.tmlcps import pseudo_outcome, tmle_continuous_treatment

N = 1200
TRUE_SLOPE = 1.5


@pytest.fixture(scope="module")
def dose():
    rng = np.random.default_rng(20260813)
    G1 = [rng.standard_normal() for _ in range(N)]
    G2 = [rng.standard_normal() for _ in range(N)]
    X = [[G1[i], G2[i]] for i in range(N)]
    D = [1.0 + 0.6 * G1[i] + 0.4 * G2[i] + rng.standard_normal()
         for i in range(N)]
    Y = [0.5 + TRUE_SLOPE * D[i] + 2.0 * G1[i] - 1.0 * G2[i]
         + 0.5 * rng.standard_normal() for i in range(N)]
    # mu broken: the outcome is quadratic in G1, the working model is not
    Ym = [0.5 + TRUE_SLOPE * D[i] + 2.0 * G1[i] * G1[i] - 1.0 * G2[i]
          + 0.5 * rng.standard_normal() for i in range(N)]
    return {"X": X, "D": D, "Y": Y, "Ym": Ym, "G1": G1}


def test_effect_curve_recovered_and_crude_is_not(dose):
    r = tmle_continuous_treatment(dose["Y"], dose["D"], dose["X"],
                                  fit="polynomial")
    assert r["estimate"] == pytest.approx(TRUE_SLOPE, abs=0.2)
    crude = kc.wls([[d] for d in dose["D"]], dose["Y"],
                   [1.0] * N)["coef"][1]
    assert abs(crude - TRUE_SLOPE) > 0.2


def test_doubly_robust_when_the_outcome_model_is_wrong(dose):
    """Theorem 1: E{xi | A=a} = theta(a) if EITHER nuisance is right.

    Breaking mu makes xi heavy-tailed, so the assertion is two standard
    errors rather than a fixed tolerance -- that is what the sampling
    distribution supports, and it can still fail.
    """
    r = tmle_continuous_treatment(dose["Ym"], dose["D"], dose["X"],
                                  fit="polynomial")
    assert abs(r["estimate"] - TRUE_SLOPE) < 2.5 * r["se"]


def test_all_three_stage_two_fits_agree(dose):
    ests = [tmle_continuous_treatment(dose["Y"], dose["D"], dose["X"],
                                      fit=f)["estimate"]
            for f in ("kernel", "locallinear", "polynomial")]
    for e in ests:
        assert e == pytest.approx(TRUE_SLOPE, abs=0.25)


def test_a_huge_bandwidth_flattens_the_curve(dose):
    """Over-smoothing shrinks the slope toward zero. If it did not, the
    bandwidth would not be doing anything."""
    r = tmle_continuous_treatment(dose["Y"], dose["D"], dose["X"],
                                  fit="kernel", bandwidth=50.0)
    assert abs(r["estimate"]) < 0.1


def test_pseudo_outcome_pieces(dose):
    xi, info = pseudo_outcome(dose["Y"], dose["D"], dose["X"])
    assert len(xi) == N
    # the marginal density is an average over everyone's covariates, so
    # it must differ from this row's own conditional density
    assert max(abs(info["marginal_density"][i] - info["pi_obs"][i])
               for i in range(N)) > 1e-6
    assert all(v > 0.0 for v in info["marginal_density"])


def test_argument_checks(dose):
    with pytest.raises(ValueError):
        tmle_continuous_treatment(
            dose["Y"], [1.0 if d > 1.0 else 0.0 for d in dose["D"]],
            dose["X"])
    with pytest.raises(ValueError):
        tmle_continuous_treatment(dose["Y"], dose["D"], dose["X"],
                                  fit="nope")
    with pytest.raises(ValueError):
        tmle_continuous_treatment(dose["Y"], dose["D"], dose["X"],
                                  bandwidth=0.0)
    with pytest.raises(ValueError):
        pseudo_outcome(dose["Y"][:-1], dose["D"], dose["X"])
