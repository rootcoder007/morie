# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.research: every argument check refuses in words (the uncovered branches of test_research.py)."""

import re

import pytest

from morie import research as R


@pytest.mark.parametrize(
    ("fn", "args", "kwargs", "msg"),
    [
        (R.dark_figure_two_source, ("x", 1, 1), {}, "single non-missing"),
        (R.dark_figure_two_source, (0, 1, 1), {}, "positive"),
        (R.dark_figure_two_source, (10, 20, 15), {}, "smaller list"),
        (R.dark_figure_two_source, (10, 20, 5), {"kappa": 0.5}, "at least 1"),
        (R.dark_figure_bounds, (1.5, 0.1, 0.1, 0.1), {}, "v_obs"),
        (R.dark_figure_bounds, (0.5, -0.1, 0.1, 0.1), {}, "r must"),
        (R.dark_figure_bounds, ([0.5, 0.6], [0.1, 0.1, 0.1], 0.1, 0.1), {}, "length 1"),
        (R.dark_figure_bounds, (0.5, 0.1, -0.1, 0.1), {}, "non-negative"),
        (R.dark_figure_bounds, (0.5, 0.1, 0.6, 0.5), {}, "below 1"),
        (R.dark_figure_breakdown, (0, 0.5), {}, "(0, 1]"),
        (R.dark_figure_breakdown, (0.5, 0.4), {}, "above v_obs"),
        (R.dark_figure_three_list, ({"100": 1},), {}, "seven cells"),
        (
            R.dark_figure_three_list,
            ({k: 0 for k in ("100", "010", "001", "110", "101", "011", "111")},),
            {},
            "positive",
        ),
        (
            R.dark_figure_three_list,
            ({k: 1 for k in ("100", "010", "001", "110", "101", "011", "111")},),
            {"candidate_missing": [0]},
            "positive",
        ),
        (R.dark_figure_hierarchy, ([0, 1],), {}, "at least one offence"),
        (R.ecological_decompose, ([1, 2], [1], ["a", "a"]), {}, "equal length"),
        (R.ecological_decompose, ([1], [1], ["a"]), {}, "at least two"),
        (R.contagion_branching, (-1,), {}, "non-negative"),
        (R.contagion_branching, (0.5,), {"mu": 0}, "positive"),
        (R.recording_map, ([[1, 0]], [1, 2]), {}, "square"),
        (R.recording_map, ([[1.2, 0], [0, 1]], [1, 2]), {}, "column sums"),
        (R.recording_map, ([[1, 0], [0, 1]], [-1, 2]), {}, "non-negative"),
        (R.detection_rate_shift, (5, 4, 1, 0.1, 0.5), {}, "counts must"),
        (R.detection_rate_shift, (1, 4, 1, 1.0, 0.5), {}, "d must"),
        (R.sentence_effect_bounds, ([1, 0], ["a"]), {}, "equal length"),
        (R.sentence_effect_bounds, ([2, 0], ["a", "b"]), {}, "0/1"),
        (R.sentence_effect_bounds, ([1, 0], ["a", "a"]), {}, "two distinct"),
        (R.sentence_effect_bounds, ([1, 0], ["a", "b"]), {"weights": [1]}, "weights"),
        (R.sentence_effect_bounds, ([1, 0], ["a", "b"]), {"contrast": "c"}, "contrast"),
        (R.contaminated_bounds, ([0.5], 1.0), {}, "[0, 1)"),
        (R.contaminated_bounds, ([1.5], 0.1), {}, "[0, 1]"),
        (R.sentence_effect_mtr, ([1, 0], ["a", "b"]), {"direction": "up"}, "direction"),
        (R.bounds_confidence, (float("nan"), 1, 1, 1), {}, "single numbers"),
        (R.bounds_confidence, (1, 0, 1, 1), {}, "at least lower"),
        (R.bounds_confidence, (0, 1, 0, 1), {}, "positive"),
        (R.bounds_confidence, (0, 1, 1, 1), {"level": 1.5}, "level"),
        (R.meta_random_effects, ([1, 2], [1]), {}, "equal length"),
        (R.meta_random_effects, ([1], [1]), {}, "at least two"),
        (R.meta_random_effects, ([1, 2], [1, -1]), {}, "positive"),
        (R.meta_random_effects, ([1, 2], [1, 1]), {"level": 2}, "level"),
        (R.meta_dl_bias, ([1],), {}, "at least two"),
        (R.meta_dl_bias, ([1, 2],), {"n_draws": 0}, "n_draws"),
        (R.meta_dl_bias, ([1, 2],), {"seed": -1}, "seed"),
        (R.cheeger_bound, ([[0, 1], [1, 0], [0, 0]], [0]), {}, "square"),
        (R.cheeger_bound, ([[0, 1], [2, 0]], [0]), {}, "symmetric"),
        (R.cheeger_bound, ([[0, 1], [1, 0]], [0, 1]), {}, "proper subset"),
        (R.cheeger_bound, ([[0, 0], [0, 0]], [0]), {}, "positive degree"),
        (R.cheeger_bound, ([[0] * 17 for _ in range(17)], [0]), {"exhaustive": True}, "degree"),
        (R.concentration_gini, ([0, 0],), {}, "positive total"),
        (R.concentration_dispersion, ([3],), {}, "positive total"),
        (R.concentration_distinct_growth, ([1],), {}, "at least two"),
        (R.fairness_rates, (0, 1, 1, 1), {}, "positive"),
        (R.fairness_implied_fpr, (1, 0.5, 0.1), {}, "p must"),
        (R.fairness_implied_fpr, (0.5, 0, 0.1), {}, "ppv"),
        (R.fairness_implied_fpr, (0.5, 0.5, 1), {}, "fnr"),
        (R.fairness_base_rate_bounds, ([1.5], 0.1, 0.1), {}, "p_obs"),
        (R.fairness_base_rate_bounds, ([0.5], -0.1, 0.1), {}, "non-negative"),
        (R.fairness_base_rate_bounds, ([0.5], 0.6, 0.5), {}, "below 1"),
        (R.fairness_true_rate, (0.5, 0.6, 0.5), {}, "alpha + beta"),
        (R.logit_rescale, (0.5, -1), {}, "non-negative"),
        (R.ranking_resolution, ([0.5], 0.1), {}, "at least two"),
        (R.ranking_resolution, ([0.5, 0.6], -0.1), {}, "non-negative"),
        (R.hazard_selection, (1.5, 0.5, 0.1, (1, 1), (1, 1)), {}, "(0, 1)"),
        (R.hazard_selection, (0.5, 0.1, 0.5, (1, 1), (1, 1)), {}, "l < h"),
        (R.hazard_selection, (0.5, 0.5, 0.1, (1,), (1, 1)), {}, "length-2"),
        (R.feedback_loop_meanfield, (0.3, 0.2, 1, 1), {"update": "other"}, "update"),
        (R.feedback_loop_meanfield, ("a", 0.2, 1, 1), {}, "lam_a must"),
        (R.feedback_loop_meanfield, (0, 0.2, 1, 1), {}, "positive"),
        (R.feedback_loop_meanfield, (0.3, 0.2, 0, 1), {}, "c_a0"),
        (R.feedback_loop_meanfield, (0.3, 0.2, 1, 1), {"rho": 2}, "rho"),
        (R.feedback_loop_meanfield, (0.3, 0.2, 1, 1), {"n_steps": -1}, "n_steps"),
        (R.feedback_loop_limit, (0.3, 0.2, 1, 1), {"update": "x"}, "update"),
        (R.feedback_loop_urn_law, (-1,), {}, "non-negative"),
        (R.feedback_loop_sim, (0.3, 0.2, 1, 1), {"update": "x"}, "update"),
        (R.feedback_loop_sim, (1.5, 0.2, 1, 1), {}, "<= 1"),
        (R.feedback_loop_sim, (0.3, 0.2, 1, 1), {"n_sims": 0}, "n_sims"),
        (R.feedback_loop_sim, (0.3, 0.2, 1, 1), {"seed": -1}, "seed"),
        (R.feedback_loop_bound, (0.2, 0.3, 1, 1), {}, "lam_a > lam_b"),
        (R.feedback_loop_bound, (0.3, 0.2, 1, 1), {"n_steps": -1}, "n_steps"),
        (R.spillover_exposure, ([1, 0], [[1, 2, 3]]), {}, "two columns"),
        (R.spillover_exposure, ([1, 0], [[1, 3]]), {}, "index places"),
        (R.spillover_effects, ([1, 2], [0], ["a", "a"]), {}, "equal length"),
        (R.spillover_effects, ([1, 2], [0, 3], ["a", "a"]), {}, "0, 1, 2"),
        (R.spillover_effects, ([1, 2], [0, 1], ["a", "a"]), {"weights": [1]}, "weights"),
        (R.spillover_ht, ([1, 2], [0, 1], [[0.5, 0.5, 0.5]]), {}, "same places"),
        (R.spillover_ht, ([1, 2], [0, 1], [[0, 0.5, 0.5], [0.5, 0.5, 0.5]]), {}, "positivity"),
        (R.spillover_ht, ([1, 2], [0, 3], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), {}, "0, 1, 2"),
        (R.spillover_ht, ([1, 2], [0, 1], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), {"joint": [[1.0]]}, "n x n x 3"),
        (R.spillover_ht_variance, ([1, 2], [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]]), {"form": "x"}, "form"),
        (R.spillover_ht_variance, ([1, 2], [0.5], [[0.5, 0.5], [0.5, 0.5]]), {}, "same places"),
        (R.spillover_ht_variance, ([1, 2], [0, 0.5], [[0.5, 0.5], [0.5, 0.5]]), {}, "positive"),
        (R.spillover_exposure_probs, (3, [[1, 2]], 3), {}, "1..n-1"),
        (R.disparity_exposure_bounds, ({"A": 1}, {"B": 1}), {}, "same groups"),
        (R.disparity_exposure_bounds, ({"A": 0}, {"A": 1}), {}, "positive"),
        (R.disparity_exposure_bounds, ({"A": 1}, {"A": 1}), {"gamma": 0.5}, ">= 1"),
        (R.disparity_exposure_bounds, ({"A": 1}, {"A": 1}), {"reference": "Z"}, "reference"),
        (R.age_crime_aggregate, ([0.5], [[1, 2]]), {}, "one share"),
        (R.age_crime_aggregate, ([0.5, 0.6], [[1, 2]]), {}, "sum to 1"),
        (R.deterrence_design_check, ([1, 2], [1], [1, 2]), {}, "equal length"),
        (R.disparity_benchmark, ({"A": 1}, {"B": 1}, {"A": 1}, "A"), {}, "same group names"),
        (R.disparity_benchmark, ({"A": 0}, {"A": 1}, {"A": 1}, "A"), {}, "positive"),
        (R.disparity_benchmark, ({"A": 1}, {"A": 1}, {"A": 1}, "Z"), {}, "reference"),
        (
            R.disparity_benchmark,
            ({"A": 1}, {"A": 1}, {"A": 1}, "A"),
            {"exposure_error_factor": {"A": 0}},
            "positive for every",
        ),
        (R.relative_risk_from_or, (0,), {}, "positive"),
        (R.relative_risk_from_or, (2,), {"base_rate": 1, "exposed_share": 0.5}, "(0, 1)"),
        (R.deterrence_response, ([1, 2], [1], [1, 2], [0.1]), {}, "equal length"),
        (R.deterrence_response, ([1, 2], [1, 2], [2, 1], [0.1]), {}, "strictly increasing"),
        (R.deterrence_response, ([1, 2], [1, 2], [1, 2], [-0.1]), {}, "non-negative"),
        (R.interracial_rates, ({"A_B": 1}, {"A": 1, "B": 1}), {}, "offender_on_victim"),
        (R.interracial_rates, ({"A_on_C": 1}, {"A": 1, "B": 1}), {}, "appear in population"),
        (R.interracial_rates, ({"A_on_B": 1}, {"A": 0, "B": 1}), {}, "positive"),
        (R.probability_of_necessity, (1.5, 0.5), {}, "[0, 1]"),
        (R.collider_arrest, (0, 0.5, 0.5), {}, "(0, 1)"),
        (R.logit_separation, ([1, 0], [[1], [1]]), {"method": "x"}, "method"),
        (R.logit_separation, ([1], [[1], [1]]), {}, "one entry per row"),
        (R.logit_separation, ([2, 0], [[1], [1]]), {}, "0/1"),
    ],
)
def test_argument_checks_refuse_in_words(fn, args, kwargs, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        fn(*args, **kwargs)


def test_remaining_branches():
    # chapman, dark figure with r = 0 (ratio NA), unidentified relative risk, the one-sided rescale on vectors
    assert R.dark_figure_two_source(400, 250, 80, chapman=True)["chapman"] > 0
    b = R.dark_figure_bounds([0.06, 0.1], 0.0, 0.01, 0.3)
    assert str(b["ratio_upper"][0]) == "nan"
    assert (
        R.probability_of_necessity(0.0, 0.0)["pn_monotone"] != R.probability_of_necessity(0.0, 0.0)["pn_monotone"]
    )  # nan
    lr = R.logit_rescale([0.5, 1.0], [0.0, 1.0])
    assert float(lr["rescale"][0]) == 1.0 and float(lr["rescale"][1]) < 1
    # ranking with a single half-width recycled
    rk = R.ranking_resolution([0.2, 0.9], 0.1)
    assert rk["share_unidentified"] == 0.0
    # ecological with a constant variable: correlations undefined
    ec = R.ecological_decompose([1, 1, 1, 1], [1, 2, 3, 4], ["a", "a", "b", "b"])
    assert ec["corr_individual"] != ec["corr_individual"]
    # concentration: a single positive place, all events on one place, every event on a new place
    assert R.concentration_decompose([0, 0, 5])["gini_positive"] == 0.0
    assert R.concentration_distinct_growth([1, 1, 1])["M_hat"] == 0.0
    assert R.concentration_distinct_growth([1, 2, 3])["M_hat"] == float("inf")
    # feedback loop: corrected update with rho, equal rates with rho, the swapped runaway
    assert R.feedback_loop_limit(0.3, 0.2, 1, 1, "corrected", rho=0.4)["share_a"] == 0.6
    assert R.feedback_loop_limit(0.2, 0.3, 1, 1)["share_a"] == 0
    assert R.feedback_loop_limit(0.3, 0.3, 1, 3)["share_a"] == 0.25
    sim = R.feedback_loop_sim(0.3, 0.2, 10, 10, n_steps=30, n_sims=2, update="corrected", rho=0.3, seed=2)
    assert sim["share_a"].shape == (2, 31)
    # spillover: Monte Carlo branch of the exposure probabilities and the single-form variances
    pr = R.spillover_exposure_probs(8, [[i, i + 1] for i in range(1, 8)], 3, n_draws=50, exact_max=1, joint=True)
    assert pr["marginal"].shape == (8, 3) and pr["joint"].shape == (8, 8, 3)
    v_ht = R.spillover_ht_variance(
        [1, 2, 3], [0.5, 0.5, 0.5], [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]
    )
    v_syg = R.spillover_ht_variance(
        [1, 2, 3], [0.5, 0.5, 0.5], [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]], form="syg"
    )
    assert isinstance(v_ht, float) and isinstance(v_syg, float)
    # HT with an unidentified variance (a joint probability of zero on a realised pair)
    ht = R.spillover_ht(
        [1, 2],
        [0, 0],
        [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
        joint=[[[0.5, 0, 0], [0, 0, 0]], [[0, 0, 0], [0.5, 0, 0]]],
    )
    assert ht["variance"]["T0"] != ht["variance"]["T0"] and ht["variance_identified"]["T0"] is False
    # recording map without names, with a 3-way three-list candidate frame
    assert R.recording_map([[1, 0], [0, 1]], [1, 2])["regime"] == "reclassification"
    # cheeger with a boolean mask
    assert R.cheeger_bound([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [True, False, False])["cut"] == 2.0
    # separation: the glm heuristic on overlapping data says none
    assert R.logit_separation([1, 0, 0, 1], [[0.1], [0.2], [0.15], [0.25]], method="glm")["separation"] == "none"
    # meta: a vector-shaped mtr direction and a non-increasing MTR
    m = R.sentence_effect_mtr([1, 0, 1, 0], ["a", "a", "b", "b"], direction="non-increasing")
    assert m["bounds"]["upper"] == 0 and m["bounds"]["lower"] <= 0
    assert R.fairness_compare_groups(0.55, 0.35, alpha_max=0.05, beta_max=0.20)["order"] == "b < a"
    agg = R.age_crime_aggregate([1.0], [[1], [2]])
    assert list(agg["type1"]) == [1.0, 2.0]
    d = R.deterrence_design_check(p=[0, 1, 0, 0], s=[0, 0, 1, 0], c=[0, 0, 0, 1])
    assert d["rank"] == 4 and all(d["identified"].values())


def test_last_branches():
    # a 1-D covariate is accepted; the glm route finds the separating direction on separable data
    assert R.logit_separation([1, 0, 0, 1], [0.1, 0.2, 0.15, 0.25])["separation"] == "none"
    g = R.logit_separation(
        [1, 1, 1, 0, 0, 0, 0], [[1, 0.2], [1, -1], [1, 0.5], [0, 0.1], [0, -0.4], [0, 1.2], [0, 0.3]], method="glm"
    )
    assert g["separation"] in ("complete", "quasi-complete") and g["direction"] is not None
    # exhaustive search refuses a connected graph with more than 16 places
    n = 17
    ring = [[1.0 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]
    with pytest.raises(ValueError, match="16 places"):
        R.cheeger_bound(ring, [0], exhaustive=True)
    # hazard selection refuses a non-positive survival probability
    with pytest.raises(ValueError, match="length-2"):
        R.hazard_selection(0.5, 0.5, 0.1, (0.0, 1.0), (1.0, 1.0))
    # a self-loop edge is ignored by the exposure mapping
    e = R.spillover_exposure([1, 0], [[1, 1], [1, 2]])
    assert list(e["exposure"]) == [2, 1]
    # the Monte Carlo branch checks its seed
    with pytest.raises(ValueError, match="seed"):
        R.spillover_exposure_probs(8, [[i, i + 1] for i in range(1, 8)], 3, n_draws=5, exact_max=1, seed=-1)


def test_recycled_half_width_and_quasi_separation():
    rk = R.ranking_resolution([0.2, 0.35, 0.8, 0.9], half_width=[0.1, 0.05])
    assert rk["n_pairs"] == 6
    # y = 1 whenever z = 1 but one y = 0 also has z = 1: the margin of that row can only be zero
    x = [[1, 0.3], [1, -0.2], [1, 0.9], [1, 0.1], [0, -0.5], [0, 0.4], [0, 1.1]]
    y = [1, 1, 1, 0, 0, 0, 0]
    r = R.logit_separation(y, x)
    assert r["separation"] in ("quasi-complete", "complete")
    assert all(m > -1e-8 for m in r["margins"]) and any(m > 1e-8 for m in r["margins"])
    assert r["n_zero_margin"] is not None
