"""Tests for smoopt, funkM, ragRet and slowdp."""
import importlib
import math

import pytest

np = importlib.import_module("morie.fn._array_core")
sm = importlib.import_module("morie.fn.smoopt")
sq = importlib.import_module("morie.fn.svmopt")
fk = importlib.import_module("morie.fn.funkM")
rr = importlib.import_module("morie.fn.ragRet")
dp = importlib.import_module("morie.fn.slowdp")

X2 = [[0.0, 0.0], [2.0, 0.0]]
Y2 = [-1.0, 1.0]
K2 = sq.kernel_matrix(X2, "linear")

RATINGS = [(0, 0, 5.0), (0, 2, 3.0), (0, 4, 4.0),
           (1, 1, 4.0), (1, 3, 2.0), (1, 5, 5.0),
           (2, 0, 4.0), (2, 3, 1.0), (2, 4, 5.0),
           (3, 1, 3.0), (3, 2, 5.0), (3, 5, 2.0),
           (4, 0, 2.0), (4, 4, 4.0), (5, 2, 4.0)]


# ---------------------------------------------------------------- smoopt
def test_smoopt_matches_the_closed_form():
    r = sm.smo_platt(Y2, K2, C=100.0)
    for a in r["alpha"]:
        assert a == pytest.approx(0.5, abs=1e-6)
    assert abs(r["equality_residual"]) < 1e-9
    assert r["kkt_violations"] == 0


def test_smoopt_and_svmopt_give_the_same_separator():
    a = sm.smo_platt(Y2, K2, C=100.0)
    b = sq.smo(Y2, K2, C=100.0)

    def f_platt(x):
        return sum(a["alpha"][j] * Y2[j] * (X2[j][0] * x)
                   for j in range(2)) - a["b"]

    def f_lib(x):
        return sum(b["alpha"][j] * Y2[j] * (X2[j][0] * x)
                   for j in range(2)) + b["b"]

    for t in (0.0, 0.5, 1.0, 2.0):
        assert f_platt(t) == pytest.approx(f_lib(t), abs=1e-6)
    assert a["b"] == pytest.approx(-b["b"], abs=1e-9)


def test_smoopt_outer_loop_alternates():
    allp = sm.outer_loop_schedule([0.0, 0.5, 1.0], 1.0, True)
    nb = sm.outer_loop_schedule([0.0, 0.5, 1.0], 1.0, False)
    assert allp["indices"] == [0, 1, 2]
    assert nb["indices"] == [1]


def test_smoopt_error_cache_and_kkt():
    E = sm.error_cache([0.0, 0.0], Y2, K2, 0.0)
    assert E[0] == pytest.approx(1.0)
    assert E[1] == pytest.approx(-1.0)
    assert sm.violates_kkt(0, [0.0, 0.0], Y2, E, 100.0)


def test_smoopt_second_choice_maximises_the_error_gap():
    rng = np.random.default_rng(0)
    p = sm.second_choice(0, [0.3, 0.4, 0.5], [-1.0, 1.0, 1.0],
                         [1.0, -3.0, 0.0], 1.0, rng)
    assert p["index"] == 1 and p["level"] == 1


def test_smoopt_rejects_bad_input():
    with pytest.raises(ValueError):
        sm.smo_platt([1.0, 0.0], K2)
    with pytest.raises(ValueError):
        sm.smo_platt(Y2, K2, C=0.0)


# ----------------------------------------------------------------- funkM
def test_funkM_mean_is_over_observed_entries():
    assert fk.global_mean(RATINGS) == pytest.approx(
        sum(r for _, _, r in RATINGS) / len(RATINGS))
    with pytest.raises(ValueError):
        fk.global_mean([])


def test_funkM_training_reduces_error_on_a_sparse_matrix():
    f = fk.fit(RATINGS, 6, 6, factors=2, epochs=300, seed=1)
    assert f["rmse"] < 0.6 * f["rmse_history"][0]
    assert f["observed"] == len(RATINGS)
    assert f["density"] < 0.5


def test_funkM_beats_imputing_the_holes():
    f = fk.fit(RATINGS, 6, 6, factors=2, epochs=300, seed=1)
    zero = fk.imputed_svd_error(RATINGS, 6, 6, rank=2, fill="zero")
    mean = fk.imputed_svd_error(RATINGS, 6, 6, rank=2, fill="mean")
    assert zero["rmse_on_observed"] > f["rmse"]
    assert mean["rmse_on_observed"] > f["rmse"]
    with pytest.raises(ValueError):
        fk.imputed_svd_error(RATINGS, 6, 6, fill="median")


def test_funkM_regularisation_shrinks_the_fit():
    tight = fk.fit(RATINGS, 6, 6, factors=2, epochs=300, seed=1,
                   reg=0.0)
    loose = fk.fit(RATINGS, 6, 6, factors=2, epochs=300, seed=1,
                   reg=0.5)
    assert tight["rmse"] < loose["rmse"]
    with pytest.raises(ValueError):
        fk.fit(RATINGS, 6, 6, reg=-1.0)


def test_funkM_incremental_is_a_different_schedule():
    joint = fk.fit(RATINGS, 6, 6, factors=2, epochs=300, seed=1)
    inc = fk.fit(RATINGS, 6, 6, factors=2, seed=1, incremental=True,
                 epochs_per_factor=150)
    assert inc["incremental"]
    assert abs(inc["rmse"] - joint["rmse"]) > 1e-9


# ---------------------------------------------------------------- ragRet
CORPUS = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [3.0, 0.2]]


def test_ragRet_metric_changes_the_answer():
    ip = rr.top_k([1.0, 0.0], CORPUS, 1, "inner_product")
    cos = rr.top_k([1.0, 0.0], CORPUS, 1, "cosine")
    assert ip["indices"] != cos["indices"]
    with pytest.raises(ValueError):
        rr.top_k([1.0, 0.0], CORPUS, 1, "euclidean")


def test_ragRet_ivf_is_approximate_and_recall_is_measured():
    corpus = [[math.cos(t * 0.7), math.sin(t * 0.7)]
              for t in range(60)]
    idx = rr.ivf_index(corpus, n_cells=6, seed=2)
    q = [1.0, 0.05]
    exact = rr.top_k(q, corpus, 5)
    near = rr.ivf_search(q, corpus, idx, 5, nprobe=1)
    allc = rr.ivf_search(q, corpus, idx, 5, nprobe=6)
    assert near["comparisons"] < 60
    assert rr.recall_at_k(near["indices"], exact["indices"])["recall"] \
        <= rr.recall_at_k(allc["indices"], exact["indices"])["recall"]
    assert rr.recall_at_k(allc["indices"],
                          exact["indices"])["recall"] == 1.0


def test_ragRet_token_composes_where_sequence_cannot():
    seq = rr.marginalise([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]],
                         "sequence")
    tok = rr.marginalise([0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]],
                         "token")
    assert tok["probability"] > 2.0 * seq["probability"]
    with pytest.raises(ValueError):
        rr.marginalise([0.5, 0.5], [[0.9, 0.1]], "token")
    with pytest.raises(ValueError):
        rr.marginalise([0.5, 0.5], [[0.9], [0.1]], "both")


def test_ragRet_rejects_degenerate_input():
    with pytest.raises(ValueError):
        rr.normalise([0.0, 0.0])
    with pytest.raises(ValueError):
        rr.top_k([1.0, 0.0], [], 1)
    with pytest.raises(ValueError):
        rr.marginalise([0.0, 0.0], [[0.5], [0.5]])


# ---------------------------------------------------------------- slowdp
def test_slowdp_tail_is_the_closed_form():
    assert dp.truncation_error(1.0, 10)["expected_tail"] == \
        pytest.approx(0.5 ** 10)
    assert dp.truncation_error(2.0, 5)["expected_tail"] == \
        pytest.approx((2.0 / 3.0) ** 5)
    with pytest.raises(ValueError):
        dp.truncation_error(0.0, 5)


def test_slowdp_simulation_matches_the_closed_form():
    n = 20000
    exact = 0.5 ** 10
    tot = sum(dp.stick_breaking(1.0, 10, seed=s)["remaining"]
              for s in range(n))
    se = math.sqrt(max(3.0 ** -10 - exact * exact, 0.0) / n)
    assert abs(tot / n - exact) < 3.0 * se


def test_slowdp_weights_and_remainder_sum_to_one():
    sb = dp.stick_breaking(1.0, 12, seed=7)
    assert sb["kept_mass"] + sb["remaining"] == pytest.approx(1.0,
                                                              abs=1e-12)


def test_slowdp_tolerance_gives_the_smallest_sufficient_K():
    r = dp.sticks_for_tolerance(1.0, 1e-3)
    assert r["expected_tail"] <= 1e-3
    assert dp.truncation_error(1.0, r["K"] - 1)["expected_tail"] > 1e-3
    assert dp.sticks_for_tolerance(5.0)["K"] > \
        dp.sticks_for_tolerance(0.5)["K"]
    with pytest.raises(ValueError):
        dp.sticks_for_tolerance(1.0, 1.5)


def test_slowdp_sticks_are_not_ordered():
    bad = sum(0 if dp.decay_diagnostics(
        dp.stick_breaking(2.0, 8, seed=s)["weights"], 2.0)["monotone"]
        else 1 for s in range(100))
    assert bad > 50


def test_slowdp_truncated_draw_reports_what_it_absorbed():
    t = dp.truncated_dp(1.0, 6, seed=3)
    assert sum(t["weights"]) == pytest.approx(1.0, abs=1e-12)
    assert t["discarded_mass"] > 0.0
    with pytest.raises(ValueError):
        dp.stick_breaking(-1.0, 5)
