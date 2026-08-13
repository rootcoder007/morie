"""Tests for tmlcds -- collaborative TMLE.

Replaces a generated test that called a stub returning mean(y). Full
anchor: ledger/wave3/anchor_tmlcds.py.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn import _s03core as kc
from morie.fn.tmlcds import ctmle_sequence, tmle_cdrs

N = 1200
TRUE = 2.0


def expit(z):
    return 1.0 / (1.0 + math.exp(-z))


@pytest.fixture(scope="module")
def instrumented():
    """W1 confounds; W2 is an instrument that predicts treatment harder
    than W1 and the outcome not at all; W3 is noise. The initial
    outcome fit sees only W3, so it is inconsistent -- the regime
    Ch. 10 is about."""
    rng = np.random.default_rng(7)
    W1 = [rng.standard_normal() for _ in range(N)]
    W2 = [rng.standard_normal() for _ in range(N)]
    W3 = [rng.standard_normal() for _ in range(N)]
    A = [1.0 if rng.uniform() < expit(-0.2 + 1.3 * W1[i] + 2.2 * W2[i])
         else 0.0 for i in range(N)]
    Y = [1.0 + TRUE * A[i] + 2.0 * W1[i] + 0.5 * rng.standard_normal()
         for i in range(N)]
    return {"Y": Y, "A": A,
            "X": [[W1[i], W2[i], W3[i]] for i in range(N)],
            "W1": W1, "W2": W2, "q": [2]}


def test_the_two_criteria_genuinely_disagree(instrumented):
    """If the instrument did not fit treatment better than the
    confounder, the collaborative check below would prove nothing."""
    d = instrumented
    g_conf = kc.treatment_density(d["A"], [[v] for v in d["W1"]])[0]
    g_inst = kc.treatment_density(d["A"], [[v] for v in d["W2"]])[0]
    assert (sum(math.log(v) for v in g_inst)
            > sum(math.log(v) for v in g_conf))


def test_sequence_is_nested_and_loss_non_increasing(instrumented):
    d = instrumented
    steps, _ = ctmle_sequence(d["Y"], d["A"], d["X"],
                              q_covariates=d["q"])
    covs = [s["covariates"] for s in steps]
    assert all(len(covs[i]) == i for i in range(len(covs)))
    assert all(set(covs[i]).issubset(set(covs[i + 1]))
               for i in range(len(covs) - 1))
    losses = [s["loss"] for s in steps]
    assert all(losses[i + 1] <= losses[i] + 1e-12
               for i in range(len(losses) - 1))


def test_confounder_chosen_over_instrument(instrumented):
    """Ch. 10's stated purpose: the instrument stays out."""
    d = instrumented
    steps, _ = ctmle_sequence(d["Y"], d["A"], d["X"],
                              q_covariates=d["q"])
    assert steps[1]["covariates"] == [0]
    losses = [s["loss"] for s in steps]
    # the confounder buys real fit; the instrument buys almost none
    assert (abs(losses[2] - losses[1])
            < 0.1 * abs(losses[1] - losses[0]))


def test_estimate_is_targeted_and_crude_is_not(instrumented):
    d = instrumented
    r = tmle_cdrs(d["Y"], d["A"], d["X"], q_covariates=d["q"])
    crude = kc.wls([[a] for a in d["A"]], d["Y"], [1.0] * N)["coef"][1]
    assert abs(crude - TRUE) > 0.5
    assert r["estimate"] == pytest.approx(TRUE, abs=0.2)
    assert r["selected_covariates"] == [0]


def test_continuous_tuning(instrumented):
    d = instrumented
    steps, _ = ctmle_sequence(d["Y"], d["A"], d["X"], tuning="continuous",
                              q_covariates=d["q"])
    crude = kc.wls([[a] for a in d["A"]], d["Y"], [1.0] * N)["coef"][1]
    # a huge penalty shrinks G to the intercept, so the estimate stays
    # near the crude one; a zero penalty does not
    assert abs(steps[0]["psi"] - crude) < abs(steps[-1]["psi"] - crude)


def test_argument_checks(instrumented):
    d = instrumented
    with pytest.raises(ValueError):
        ctmle_sequence(d["Y"], d["A"], d["X"], q_covariates=[99])
    with pytest.raises(ValueError):
        ctmle_sequence(d["Y"], [2.0] * N, d["X"])
    with pytest.raises(ValueError):
        ctmle_sequence(d["Y"], d["A"], d["X"], tuning="nope")
