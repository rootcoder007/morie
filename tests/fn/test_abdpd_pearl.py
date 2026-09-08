"""Abduction-action-prediction, anchored on Pearl (2000) Section 1.4.

The worked example (model 2 of eq. (1.48), pp. 36-37, the copy held in the
corpus): x = u1, y = 1 iff x == u2, both exogenous variables binary.
Evidence {y=1, x=1} is "compatible with only one realization of U1 and U2
-- namely u1 = 1, u2 = 1", and under do(x=0) "the probability of recovery
(y = 0) is unity".
"""

import warnings

import pytest

warnings.filterwarnings("ignore")

from morie.fn.abdpd import abduction_modification_prediction as aap

BINARY_EQS = {
    "x": (("u1",), lambda u1: u1),
    "y": (("x", "u2"), lambda x, u2: 1.0 if abs(x - u2) < 0.5 else 0.0),
}


def test_pearl_1_4_worked_example():
    r = aap(evidence={"x": 1.0, "y": 1.0}, equations=BINARY_EQS,
            exogenous_names=["u1", "u2"], do={"x": 0.0}, query="y")
    assert r["abducted"] == {"u1": 1.0, "u2": 1.0}
    assert r["factual"] == 1.0
    assert r["counterfactual"] == 0.0
    assert r["counterfactual_unique"] is True
    assert r["n_compatible_u"] == 1
    assert r["residual"] == 0.0


def test_pearl_other_cell_of_the_same_model():
    # a treated survivor (x=1, y=0) must have u2=0: dies iff NOT treated,
    # so under do(x=0) this subject would have died (y=1)
    r = aap(evidence={"x": 1.0, "y": 0.0}, equations=BINARY_EQS,
            exogenous_names=["u1", "u2"], do={"x": 0.0}, query="y")
    assert r["abducted"] == {"u1": 1.0, "u2": 0.0}
    assert r["counterfactual"] == 1.0


def test_ambiguous_evidence_is_reported_not_hidden():
    # observing only x=1 leaves u2 free: both u2 values reproduce the
    # evidence and they give DIFFERENT counterfactuals for y, so the
    # result must say so rather than pick one silently
    r = aap(evidence={"x": 1.0}, equations=BINARY_EQS,
            exogenous_names=["u1", "u2"], do={"x": 0.0}, query="y")
    assert r["n_compatible_u"] == 2
    assert r["counterfactual_unique"] is False


def test_continuous_model_still_uses_the_gradient_path():
    # linear Gaussian-style SCM: abduction is exactly solvable and the
    # counterfactual is the classic shift formula
    eqs = {
        "x": (("u1",), lambda u1: u1),
        "y": (("x", "u2"), lambda x, u2: 2.0 * x + u2),
    }
    r = aap(evidence={"x": 1.0, "y": 3.5}, equations=eqs,
            exogenous_names=["u1", "u2"], do={"x": 2.0}, query="y")
    assert abs(r["abducted"]["u1"] - 1.0) < 1e-8
    assert abs(r["abducted"]["u2"] - 1.5) < 1e-8
    assert abs(r["counterfactual"] - 5.5) < 1e-8
    assert "gradient" in r["method"]


def test_support_too_large_is_refused():
    def _mk(name):
        # scmdf calls equations with keyword arguments named after the
        # parents; a STEP function keeps the gradient solver stuck so the
        # discrete path is genuinely reached
        return lambda **kw: 1.0 if kw[name] > 0.5 else 0.0

    eqs = {"y%d" % i: (("u%d" % i,), _mk("u%d" % i)) for i in range(30)}
    with pytest.raises(ValueError):
        # declaring the support routes straight to the discrete path,
        # whose size guard refuses 2^30 candidates immediately
        aap(evidence={"y%d" % i: 7.7 for i in range(30)}, equations=eqs,
            exogenous_names=["u%d" % i for i in range(30)],
            do={"y0": 0.0}, query="y1", u_support=(0.0, 1.0))
