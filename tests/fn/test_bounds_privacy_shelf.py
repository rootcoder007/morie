"""Permutation LM loss, the private mean, and the bounds no estimator beats.

Sources: Yang et al (2019) arXiv:1906.08237 (XLNet); Dwork and Roth
(2014) Sec 3.3 and Thm A.1; Hahn (1998) *Econometrica* 66:315-331;
Hirano and Porter (2009) *Econometrica* 77:1683-1701.
"""

import math

import numpy as np
import pytest

from morie.fn.bnsadm import (
    bound_admissible_estimators,
    minimax_regret_constant,
)
from morie.fn.gestee import dp_mean_error_curve, gauss_subgaussian_estimator
from morie.fn.kmperm import (
    kamath_permutation_lm_loss,
    permutation_attention_masks,
)


# --------------------------------------------------------------------
# XLNet permutation language model
# --------------------------------------------------------------------

def toy_logits(seed=0, T=8, V=12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, V)), rng.integers(0, V, size=T)


def test_loss_is_the_average_negative_log_likelihood():
    lg = np.log([[0.5, 0.5], [0.25, 0.75]])
    out = kamath_permutation_lm_loss(lg, [0, 1], [0, 1])
    assert out["loss"] == pytest.approx((-math.log(0.5) - math.log(0.75)) / 2)


def test_the_full_sequence_loss_does_not_depend_on_the_permutation():
    # reordering the terms of a sum does not change the sum; the
    # permutation acts on the model's conditioning, not on this
    # arithmetic, and mistaking the two is the usual misreading
    lg, y = toy_logits()
    rng = np.random.default_rng(1)
    base = kamath_permutation_lm_loss(lg, y, np.arange(8))["loss"]
    for _ in range(10):
        z = rng.permutation(8)
        assert kamath_permutation_lm_loss(lg, y, z)["loss"] == pytest.approx(
            base, abs=1e-12
        )


def test_order_invariance_is_reported_and_partial_prediction_breaks_it():
    lg, y = toy_logits()
    assert kamath_permutation_lm_loss(lg, y, np.arange(8))["order_invariant"]
    part = kamath_permutation_lm_loss(lg, y, np.arange(8), num_predict=3)
    assert part["order_invariant"] is False
    assert part["partial_prediction"] is True
    assert any("factorization order" in w for w in part.warnings)


def test_partial_prediction_actually_differs_across_orders():
    lg, y = toy_logits(seed=2)
    a = kamath_permutation_lm_loss(lg, y, np.arange(8), num_predict=2)["loss"]
    b = kamath_permutation_lm_loss(lg, y, np.arange(8)[::-1],
                                   num_predict=2)["loss"]
    assert a != pytest.approx(b)


def test_partial_prediction_scores_the_trailing_positions_in_order():
    lg, y = toy_logits(seed=3)
    z = np.array([5, 2, 7, 0, 3, 1, 6, 4])
    out = kamath_permutation_lm_loss(lg, y, z, num_predict=3)
    assert list(out["scored_positions"]) == [1, 6, 4]


def test_partial_prediction_scores_the_longest_contexts():
    # eq (5) keeps the tail of the factorization order precisely because
    # those positions condition on the most tokens
    lg, y = toy_logits(seed=4)
    z = np.arange(8)
    full = kamath_permutation_lm_loss(lg, y, z)["mean_context_length"]
    tail = kamath_permutation_lm_loss(lg, y, z,
                                      num_predict=2)["mean_context_length"]
    assert tail > full


def test_reduction_sum_and_none():
    lg, y = toy_logits(seed=5)
    z = np.arange(8)
    s = kamath_permutation_lm_loss(lg, y, z, reduction="sum")["loss"]
    m = kamath_permutation_lm_loss(lg, y, z, reduction="mean")["loss"]
    per = kamath_permutation_lm_loss(lg, y, z, reduction="none")["loss"]
    assert s == pytest.approx(m * 8)
    assert per.shape == (8,)
    assert float(np.sum(per)) == pytest.approx(s)


def test_perplexity_is_the_exponential_of_the_mean_loss():
    lg, y = toy_logits(seed=6)
    out = kamath_permutation_lm_loss(lg, y, np.arange(8))
    assert out["perplexity"] == pytest.approx(math.exp(out["loss"]))


def test_a_uniform_model_has_perplexity_equal_to_the_vocabulary_size():
    V = 17
    lg = np.zeros((5, V))
    out = kamath_permutation_lm_loss(lg, [0, 1, 2, 3, 4], np.arange(5))
    assert out["perplexity"] == pytest.approx(V)


def test_log_softmax_is_stable_at_extreme_logits():
    lg = np.array([[1e4, 0.0], [-1e4, 0.0]])
    out = kamath_permutation_lm_loss(lg, [0, 1], [0, 1])
    assert np.all(np.isfinite(out["token_nll"]))
    assert out["token_nll"][0] == pytest.approx(0.0, abs=1e-9)


def test_the_two_masks_differ_exactly_on_the_diagonal():
    # the content stream sees the token at its own position and the
    # query stream must not; that single bit is why XLNet needs two
    m = permutation_attention_masks([2, 0, 3, 1])
    assert np.array_equal(m["content"] ^ m["query"], np.eye(4, dtype=bool))


def test_the_identity_permutation_gives_the_causal_mask():
    m = permutation_attention_masks(np.arange(6))
    assert np.array_equal(m["content"], np.tril(np.ones((6, 6), bool)))


def test_the_reversed_permutation_gives_the_anti_causal_mask():
    m = permutation_attention_masks(np.arange(6)[::-1])
    assert np.array_equal(m["content"], np.triu(np.ones((6, 6), bool)))


def test_each_position_attends_to_exactly_its_rank_many_predecessors():
    m = permutation_attention_masks([3, 1, 4, 0, 2])
    assert list(m["query"].sum(axis=1)) == list(m["rank"])


def test_averaging_over_permutations_makes_the_context_bidirectional():
    # the point of eq (3): no single order sees both sides, but the
    # expectation over orders does
    T = 6
    rng = np.random.default_rng(0)
    seen = np.zeros((T, T), dtype=bool)
    for _ in range(500):
        seen |= permutation_attention_masks(rng.permutation(T))["query"]
    off = ~np.eye(T, dtype=bool)
    assert seen[off].all()


def test_permutation_validation():
    lg, y = toy_logits()
    with pytest.raises(ValueError, match="must be a permutation"):
        kamath_permutation_lm_loss(lg, y, [0, 0, 1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="must not be empty"):
        permutation_attention_masks([])


def test_input_validation():
    lg, y = toy_logits()
    with pytest.raises(ValueError, match="targets has length"):
        kamath_permutation_lm_loss(lg, y[:3], np.arange(8))
    with pytest.raises(ValueError, match="targets must lie"):
        kamath_permutation_lm_loss(lg, np.full(8, 99), np.arange(8))
    with pytest.raises(ValueError, match="num_predict"):
        kamath_permutation_lm_loss(lg, y, np.arange(8), num_predict=0)
    with pytest.raises(ValueError, match="reduction"):
        kamath_permutation_lm_loss(lg, y, np.arange(8), reduction="avg")


# --------------------------------------------------------------------
# Differentially private mean
# --------------------------------------------------------------------

def test_sensitivity_is_the_width_over_n():
    out = gauss_subgaussian_estimator(np.arange(50.0), C=10.0, lower=0.0,
                                      epsilon=1.0, seed=0)
    assert out["sensitivity"] == pytest.approx(10.0 / 50)


def test_the_mechanism_is_unbiased_about_the_clipped_mean():
    x = np.linspace(0.0, 1.0, 200)
    draws = [gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=1.0,
                                         seed=s)["estimate"]
             for s in range(4000)]
    mu = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=1.0,
                                     seed=0)["clipped_mean"]
    assert abs(float(np.mean(draws)) - mu) < 0.002


def test_the_noise_standard_deviation_matches_the_laplace_scale():
    x = np.linspace(0.0, 1.0, 100)
    draws = np.array([
        gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=0.5,
                                    seed=s)["noise_drawn"]
        for s in range(8000)
    ])
    expected = math.sqrt(2.0) * (1.0 / 100) / 0.5
    assert float(np.std(draws)) == pytest.approx(expected, rel=0.05)


def test_tighter_privacy_costs_more_noise():
    x = np.linspace(0.0, 1.0, 100)
    a = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=0.1, seed=0)
    b = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=10.0, seed=0)
    assert a["noise_sd"] > 10 * b["noise_sd"]


def test_more_data_costs_less_noise_at_a_fixed_epsilon():
    a = gauss_subgaussian_estimator(np.linspace(0, 1, 100), C=1.0, lower=0.0,
                                    epsilon=1.0, seed=0)
    b = gauss_subgaussian_estimator(np.linspace(0, 1, 10000), C=1.0,
                                    lower=0.0, epsilon=1.0, seed=0)
    assert a["noise_sd"] == pytest.approx(100 * b["noise_sd"], rel=1e-9)


def test_clipping_bias_is_reported_and_signed_correctly():
    x = np.array([0.0, 0.0, 0.0, 100.0])
    out = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=1.0,
                                      seed=0)
    assert out["n_clipped"] == 1
    assert out["clipped_mean"] < out["non_private_mean"]
    assert out["clipping_bias"] < 0
    assert any("clipped" in w for w in out.warnings)


def test_clipping_bias_is_zero_when_nothing_is_clipped():
    x = np.linspace(0.2, 0.8, 50)
    out = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=1.0,
                                      seed=0)
    assert out["n_clipped"] == 0
    assert out["clipping_bias"] == pytest.approx(0.0, abs=1e-12)


def test_choosing_the_width_from_the_data_is_flagged_as_a_leak():
    out = gauss_subgaussian_estimator(np.linspace(0, 1, 40), epsilon=1.0,
                                      seed=0)
    assert any("range of y" in w for w in out.warnings)


def test_the_naive_interval_is_narrower_than_the_honest_one():
    x = np.linspace(0.0, 1.0, 60)
    out = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=0.2,
                                      seed=0)
    naive_w = out["ci_naive_upper"] - out["ci_naive_lower"]
    honest_w = out["ci_upper"] - out["ci_lower"]
    assert honest_w > naive_w


def test_the_honest_interval_covers_and_the_naive_one_undercovers():
    # the whole point of separating them: ignoring the mechanism noise
    # is not a rounding error at a small epsilon
    x = np.linspace(0.0, 1.0, 200)
    truth = float(np.mean(x))
    hit_h = hit_n = 0
    reps = 1500
    for s in range(reps):
        o = gauss_subgaussian_estimator(x, C=1.0, lower=0.0, epsilon=0.05,
                                        seed=s)
        hit_h += o["ci_lower"] <= truth <= o["ci_upper"]
        hit_n += o["ci_naive_lower"] <= truth <= o["ci_naive_upper"]
    assert hit_h / reps > 0.95
    assert hit_n / reps < 0.60


def test_the_gaussian_mechanism_uses_the_dwork_roth_sigma():
    out = gauss_subgaussian_estimator(np.linspace(0, 1, 100), C=1.0,
                                      lower=0.0, epsilon=0.5,
                                      mechanism="gaussian", delta=1e-6,
                                      seed=0)
    expected = (1.0 / 100) * math.sqrt(2 * math.log(1.25 / 1e-6)) / 0.5
    assert out["noise_scale"] == pytest.approx(expected)


def test_the_gaussian_bound_warns_outside_its_proved_range():
    out = gauss_subgaussian_estimator(np.linspace(0, 1, 50), C=1.0,
                                      lower=0.0, epsilon=2.0,
                                      mechanism="gaussian", seed=0)
    assert any("epsilon < 1" in w for w in out.warnings)


WIDTHS = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 60.0, 200.0])


def test_the_error_curve_has_an_interior_minimum_when_the_data_are_skewed():
    rng = np.random.default_rng(0)
    x = rng.exponential(1.0, size=4000)
    out = dp_mean_error_curve(x, WIDTHS, epsilon=0.05, reps=800, seed=1)
    assert out["interior_minimum"] is True
    # the exact minimiser sits at 4 or 8 depending on the draw -- their
    # RMSEs differ by under 5 % -- so pin the shape, not the argmin
    assert out["best_rmse"] < out["rmse"][0]
    assert out["best_rmse"] < out["rmse"][-1]
    assert out["max_abs_bias"] > 0.05
    assert np.all(np.diff(out["noise_sd"]) > 0)


def test_a_centred_window_on_symmetric_data_has_no_trade_off_at_all():
    # the bias cancels between the two tails at every width, so the
    # RMSE is monotone in C and the smallest width simply wins; the
    # usual "bias against noise" story does not apply here
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    out = dp_mean_error_curve(x, WIDTHS, epsilon=0.05, reps=800, seed=1)
    assert out["max_abs_bias"] < 0.01
    assert out["interior_minimum"] is False
    assert out["best_width"] == 0.5
    assert np.all(np.diff(out["rmse"]) > 0)
    assert any("no trade-off" in w for w in out.warnings)


def test_bias_need_not_be_monotone_in_the_width():
    # on a lognormal sample the sliding window's lower edge leaves the
    # support before its upper edge covers the tail, so the bias gets
    # worse before it gets better
    rng = np.random.default_rng(0)
    x = rng.lognormal(0.0, 1.0, size=4000)
    out = dp_mean_error_curve(x, WIDTHS, epsilon=0.05, reps=200, seed=1)
    b = np.abs(out["bias"])
    assert np.any(np.diff(b) > 0) and np.any(np.diff(b) < 0)


def test_dp_input_validation():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        gauss_subgaussian_estimator([1.0, 2.0], C=1.0, epsilon=0.0)
    with pytest.raises(ValueError, match="C must be positive"):
        gauss_subgaussian_estimator([1.0, 2.0], C=-1.0)
    with pytest.raises(ValueError, match="mechanism"):
        gauss_subgaussian_estimator([1.0, 2.0], C=1.0, mechanism="cauchy")
    with pytest.raises(ValueError, match="at least one finite"):
        gauss_subgaussian_estimator([np.nan], C=1.0)
    with pytest.raises(ValueError, match="delta"):
        gauss_subgaussian_estimator([1.0, 2.0], C=1.0, mechanism="gaussian",
                                    delta=2.0)


# --------------------------------------------------------------------
# Efficiency and minimax bounds
# --------------------------------------------------------------------

def ate_design(seed=0, n=4000, tau=1.0, conf=0.6, het=0.0, sd=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    e = 1.0 / (1.0 + np.exp(-conf * X[:, 0]))
    D = (rng.random(n) < e).astype(float)
    tau_x = tau + het * X[:, 1]
    y = 2.0 + X @ np.array([1.0, -0.5]) + D * tau_x + rng.normal(size=n) * sd
    return y, D, X, e, tau_x


def test_the_minimax_constant_solves_its_own_stationarity_condition():
    mc = minimax_regret_constant()
    assert mc["stationarity_residual"] < 1e-12
    # solved, not quoted: the two-figure 0.17 in the literature is not
    # precise enough to check an attainment claim against simulation
    assert mc["constant"] == pytest.approx(0.16997120747990366, abs=1e-12)
    assert mc["t_star"] == pytest.approx(0.7518, abs=1e-3)


def test_the_constant_really_is_the_maximum():
    mc = minimax_regret_constant()
    t = np.linspace(0.0, 4.0, 40001)
    grid = t * (0.5 * np.array([math.erfc(v / math.sqrt(2)) for v in t]))
    assert float(np.max(grid)) <= mc["constant"] + 1e-9


def test_aipw_recovers_the_treatment_effect():
    y, D, X, _, _ = ate_design()
    out = bound_admissible_estimators(y, D, X)
    assert abs(out["estimate"] - 1.0) < 0.1


def test_aipw_attains_the_bound_and_ipw_does_not():
    # the defining property: an efficient estimator sits at the bound,
    # an inefficient consistent one sits above it
    y, D, X, _, _ = ate_design(n=8000)
    out = bound_admissible_estimators(y, D, X)
    assert out["aipw_efficiency_ratio"] == pytest.approx(1.0, abs=0.1)
    assert out["ipw_efficiency_ratio"] > 1.3


def test_the_empirical_variance_of_aipw_matches_the_bound():
    # 600 replications, not a hundred: the sampling distribution of a
    # variance is chi-square, so a short run reads several tens of a
    # percent low often enough to look like a real violation of the
    # bound. Averaging the bound over the same replications matters
    # too -- a single realisation of V_eff is itself noisy.
    n, reps = 1500, 600
    est, veff = [], []
    for s in range(reps):
        out = bound_admissible_estimators(*ate_design(seed=200 + s, n=n)[:3])
        est.append(out["estimate"])
        veff.append(out["efficiency_bound"])
    emp = float(np.var(est, ddof=1))
    bound = float(np.mean(veff)) / n
    assert emp == pytest.approx(bound, rel=0.15)
    assert emp >= bound * 0.9


def test_worse_overlap_raises_the_bound():
    a = bound_admissible_estimators(*ate_design(conf=0.2)[:3])
    b = bound_admissible_estimators(*ate_design(conf=2.5)[:3])
    assert b["efficiency_bound"] > a["efficiency_bound"]
    assert b["overlap_term"] > a["overlap_term"]


def test_heterogeneity_adds_to_the_bound_and_is_isolated():
    a = bound_admissible_estimators(*ate_design(het=0.0)[:3])
    b = bound_admissible_estimators(*ate_design(het=1.5)[:3])
    assert a["heterogeneity_term"] < 0.05
    assert b["heterogeneity_term"] > 1.0
    assert b["efficiency_bound"] > a["efficiency_bound"]


def test_the_two_components_sum_to_the_bound():
    out = bound_admissible_estimators(*ate_design(het=0.8)[:3])
    assert out["overlap_term"] + out["heterogeneity_term"] == pytest.approx(
        out["efficiency_bound"]
    )


def test_a_noisier_outcome_raises_only_the_overlap_term():
    a = bound_admissible_estimators(*ate_design(sd=0.5)[:3])
    b = bound_admissible_estimators(*ate_design(sd=2.0)[:3])
    assert b["overlap_term"] > 3 * a["overlap_term"]
    assert abs(b["heterogeneity_term"] - a["heterogeneity_term"]) < 0.05


def test_the_bound_on_the_standard_error_scales_as_one_over_root_n():
    a = bound_admissible_estimators(*ate_design(seed=1, n=1000)[:3])
    b = bound_admissible_estimators(*ate_design(seed=1, n=16000)[:3])
    assert a["se_bound"] / b["se_bound"] == pytest.approx(4.0, rel=0.2)


def test_the_regret_bound_is_the_constant_times_the_se_bound():
    out = bound_admissible_estimators(*ate_design()[:3])
    assert out["minimax_regret_bound"] == pytest.approx(
        out["minimax_constant"] * out["se_bound"]
    )


def test_the_plug_in_rule_does_not_beat_the_minimax_regret_bound():
    # simulate the local experiment the bound is stated for: the rule
    # treats when the estimate is positive, and its worst-case regret
    # over the local parameter cannot fall below c * sqrt(V/n)
    mc = minimax_regret_constant()
    V, n = 4.0, 2000
    rng = np.random.default_rng(0)
    worst = 0.0
    for h in np.linspace(0.0, 6.0, 61):
        tau = h / math.sqrt(n)
        est = rng.normal(tau, math.sqrt(V / n), size=40000)
        regret = float(np.mean(np.where(est > 0, 0.0, max(tau, 0.0))))
        worst = max(worst, regret)
    bound = mc["constant"] * math.sqrt(V / n)
    assert worst <= bound * 1.05
    assert worst > bound * 0.9


def test_a_binary_outcome_uses_the_bernoulli_variance():
    rng = np.random.default_rng(3)
    n = 6000
    X = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    D = (rng.random(n) < e).astype(float)
    p = 1 / (1 + np.exp(-(0.3 * X[:, 0] + 0.8 * D)))
    y = (rng.random(n) < p).astype(float)
    out = bound_admissible_estimators(y, D, X, family="binomial")
    assert 0.0 < out["estimate"] < 0.4
    assert np.all(out["mu1"] > 0) and np.all(out["mu1"] < 1)


def test_trimming_is_reported_when_it_binds():
    y, D, X, _, _ = ate_design(conf=4.0, n=3000)
    out = bound_admissible_estimators(y, D, X, trim=0.05)
    assert out["trim_binding"] > 0
    assert any("trimmed" in w for w in out.warnings)


def test_no_trimming_warning_under_good_overlap():
    out = bound_admissible_estimators(*ate_design(conf=0.2)[:3], trim=0.01)
    assert out["trim_binding"] == 0
    assert not any("trimmed" in w for w in out.warnings)


def test_bounds_input_validation():
    y, D, X, _, _ = ate_design(n=200)
    with pytest.raises(ValueError, match="must agree in length"):
        bound_admissible_estimators(y[:100], D, X)
    with pytest.raises(ValueError, match="D must be binary"):
        bound_admissible_estimators(y, D * 2, X)
    with pytest.raises(ValueError, match="family"):
        bound_admissible_estimators(y, D, X, family="poisson")
    with pytest.raises(ValueError, match="binary outcome"):
        bound_admissible_estimators(y, D, X, family="binomial")
    with pytest.raises(ValueError, match="at least 10"):
        bound_admissible_estimators(y[:5], D[:5], X[:5])
