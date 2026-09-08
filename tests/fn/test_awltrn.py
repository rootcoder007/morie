"""awltrn -- augmented outcome-weighted learning. Source: Liu, Y.,
Wang, Y., Kosorok, M. R., Zhao, Y. & Zeng, D. (2018) Statistics in
Medicine, doi:10.1002/sim.7844."""
import pytest

from morie.fn import _array_core as np
from morie.fn.awltrn import (aol_weights, fit_aol, fit_stages,
                             owl_weights, regimen_value,
                             weighted_rule)


def trial(n=600, seed=7, prognosis=20.0):
    rng = np.random.default_rng(seed)
    R, A, H = [], [], []
    for _ in range(n):
        x0 = float(rng.normal())
        x1 = float(rng.normal())
        a = 1 if float(rng.uniform()) < 0.5 else -1
        R.append(prognosis + 5.0 * x1
                 + a * (1.0 if x0 > 0.0 else -1.0)
                 + float(rng.normal(0.0, 0.5)))
        A.append(a)
        H.append([x0, x1])
    return R, A, H


def oracle(x):
    return 1 if x[0] > 0.0 else -1


def test_owl_weights_are_non_negative():
    R, A, H = trial()
    assert all(v >= 0.0 for v in owl_weights(R, A, H)["weights"])


def test_owl_keeps_the_observed_labels():
    R, A, H = trial()
    assert owl_weights(R, A, H)["labels"] == A


def test_aol_weights_are_smaller_than_owl_weights():
    R, A, H = trial()
    a = sum(aol_weights(R, A, H)["weights"])
    o = sum(owl_weights(R, A, H)["weights"])
    assert a < 0.2 * o


def test_aol_flips_labels_on_negative_residuals():
    R, A, H = trial()
    w = aol_weights(R, A, H)
    assert 0 < w["n_flipped"] < len(A)
    for i in range(len(A)):
        if w["residual"][i] < 0.0:
            assert w["labels"][i] == -A[i]


def test_aol_recovers_the_planted_rule():
    R, A, H = trial()
    f = fit_aol(R, A, H, propensity=0.5)
    acc = sum(1 for i in range(len(H))
              if f["rule"](H[i]) == oracle(H[i])) / len(H)
    assert acc > 0.85


def test_aol_beats_owl_when_prognosis_dominates():
    R, A, H = trial(prognosis=50.0)
    a = fit_aol(R, A, H, propensity=0.5, method="aol")
    o = fit_aol(R, A, H, propensity=0.5, method="owl")
    acc_a = sum(1 for i in range(len(H))
                if a["rule"](H[i]) == oracle(H[i])) / len(H)
    acc_o = sum(1 for i in range(len(H))
                if o["rule"](H[i]) == oracle(H[i])) / len(H)
    assert acc_a > acc_o


def test_aol_is_invariant_to_shifting_the_outcome():
    R, A, H = trial()
    a = fit_aol(R, A, H, propensity=0.5)
    b = fit_aol([v - 100.0 for v in R], A, H, propensity=0.5)
    acc_a = sum(1 for i in range(len(H))
                if a["rule"](H[i]) == oracle(H[i])) / len(H)
    acc_b = sum(1 for i in range(len(H))
                if b["rule"](H[i]) == oracle(H[i])) / len(H)
    assert acc_a == pytest.approx(acc_b, abs=0.02)


def test_owl_refuses_an_insufficient_shift():
    R, A, H = trial()
    with pytest.raises(ValueError):
        owl_weights([v - 100.0 for v in R], A, H, shift=0.0)


def test_a_misspecified_prognostic_model_still_returns_a_rule():
    R, A, H = trial()
    f = fit_aol(R, A, H, propensity=0.5, prognostic=[0.0] * len(R))
    assert callable(f["rule"])
    assert f["value"] > 0.0


def test_regimen_value_of_the_oracle_beats_a_constant_rule():
    R, A, H = trial()
    v_or = regimen_value(R, A, H, oracle, propensity=0.5)
    v_1 = regimen_value(R, A, H, lambda x: 1, propensity=0.5)
    assert v_or > v_1


def test_weighted_rule_returns_a_callable_and_coefficients():
    R, A, H = trial()
    w = aol_weights(R, A, H)
    c = weighted_rule(H, w["labels"], w["weights"])
    assert callable(c["rule"])
    assert len(c["coef"]) == 3


def test_multi_stage_uses_every_subject_at_every_stage():
    R, A, H = trial()
    s = fit_stages([(R, A, H), (R, A, H)], propensity=0.5)
    assert s["n_stages"] == 2
    assert all(v == len(R) for v in s["n_used_per_stage"])


def test_a_wrong_treatment_coding_is_refused():
    R, A, H = trial()
    with pytest.raises(ValueError):
        aol_weights(R, [0] * len(A), H)


def test_a_degenerate_propensity_is_refused():
    R, A, H = trial()
    with pytest.raises(ValueError):
        aol_weights(R, A, H, propensity=1.0)


def test_an_unknown_method_is_refused():
    R, A, H = trial()
    with pytest.raises(ValueError):
        fit_aol(R, A, H, method="qlearning")


def test_mismatched_lengths_are_refused():
    R, A, H = trial()
    with pytest.raises(ValueError):
        aol_weights(R, A[:-1], H)


def test_no_stages_is_refused():
    with pytest.raises(ValueError):
        fit_stages([])


def test_stages_of_differing_size_are_refused():
    R, A, H = trial()
    with pytest.raises(ValueError):
        fit_stages([(R, A, H), (R[:-1], A[:-1], H[:-1])])
