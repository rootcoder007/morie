"""trnsfr -- transporting an effect between cohorts.
Source: Wager (2025) Causal Inference: A Statistical Learning
Approach, chs. 2, 3 and 7."""
import pytest

from morie.fn.trnsfr import (balancing_weights, transfer_msm,
                             transport_ate, transport_weights)


def cohorts(n=200, seed=0):
    from morie.fn import _array_core as np
    rng = np.random.default_rng(seed)
    X, S, W, Y = [], [], [], []
    tau = {0: 1.0, 1: 5.0}
    for i in range(n):
        src = 1.0 if i < n // 2 else 0.0
        p1 = 0.25 if src == 1.0 else 0.75
        x = 1.0 if float(rng.uniform()) < p1 else 0.0
        w = 1.0 if float(rng.uniform()) < 0.5 else 0.0
        X.append([x])
        S.append(src)
        W.append(w)
        Y.append(x + w * tau[int(x)] + float(rng.normal(0.0, 0.1)))
    return Y, W, X, S


def test_transport_weights_are_positive_only_on_the_source():
    _, _, X, S = cohorts()
    w = transport_weights(X, S)["weights"]
    assert all(w[i] == 0.0 for i in range(len(S)) if S[i] == 0.0)
    assert all(w[i] > 0.0 for i in range(len(S)) if S[i] == 1.0)


def test_transport_weights_average_to_one_on_the_source():
    _, _, X, S = cohorts()
    d = transport_weights(X, S)
    ws = [d["weights"][i] for i in range(len(S)) if S[i] == 1.0]
    assert sum(ws) / len(ws) == pytest.approx(1.0, abs=1e-9)


def test_effective_sample_size_never_exceeds_the_source_size():
    _, _, X, S = cohorts()
    d = transport_weights(X, S)
    assert d["ess"] <= d["n_source"] + 1e-9


def test_balancing_weights_match_the_target_moments_exactly():
    _, _, X, S = cohorts()
    b = balancing_weights(X, S)
    assert b["max_imbalance"] < 1e-8


def test_balancing_weights_balance_under_a_wrong_functional_form():
    _, _, X, S = cohorts()
    Xq = [[r[0], r[0] ** 3] for r in X]
    assert balancing_weights(Xq, S)["max_imbalance"] < 1e-8


def test_all_four_routes_move_off_the_source_ate():
    Y, W, X, S = cohorts(400, seed=2)
    src = transport_ate(Y, W, X, S, method="dr", e=0.5)["source_ate"]
    for m in ("ipw", "outcome", "dr", "balance"):
        est = transport_ate(Y, W, X, S, method=m, e=0.5)["estimate"]
        assert abs(est - src) > 0.5


def test_dr_and_outcome_agree_when_the_outcome_model_is_saturated():
    Y, W, X, S = cohorts(400, seed=3)
    a = transport_ate(Y, W, X, S, method="dr", e=0.5)["estimate"]
    b = transport_ate(Y, W, X, S, method="outcome", e=0.5)["estimate"]
    assert a == pytest.approx(b, abs=0.15)


def test_transfer_msm_returns_a_two_element_coefficient_vector():
    Y, W, X, S = cohorts(400, seed=4)
    lab = ["src" if v == 1.0 else "tgt" for v in S]
    r = transfer_msm(Y, W, X, lab, target="tgt", e=0.5)
    assert len(r["coef"]) == 2
    assert r["estimate"] == pytest.approx(r["coef"][1], abs=1e-15)


def test_non_binary_cohort_indicator_is_refused():
    _, _, X, _ = cohorts()
    with pytest.raises(ValueError):
        transport_weights(X, [0.5] * len(X))


def test_a_cohort_with_fewer_than_two_units_is_refused():
    _, _, X, S = cohorts()
    bad = [1.0] * (len(S) - 1) + [0.0]
    with pytest.raises(ValueError):
        transport_weights(X, bad)


def test_absent_overlap_is_refused():
    _, _, _, S = cohorts()
    X = [[1.0] if v == 1.0 else [-20.0] for v in S]
    with pytest.raises(ValueError):
        transport_weights(X, S)


def test_unknown_method_is_refused():
    Y, W, X, S = cohorts()
    with pytest.raises(ValueError):
        transport_ate(Y, W, X, S, method="bayes")


def test_absent_target_cohort_is_refused():
    Y, W, X, S = cohorts()
    with pytest.raises(ValueError):
        transfer_msm(Y, W, X, ["a"] * len(S), target="b")


def test_propensity_outside_the_open_unit_interval_is_refused():
    Y, W, X, S = cohorts()
    with pytest.raises(ValueError):
        transport_ate(Y, W, X, S, method="ipw", e=0.0)
