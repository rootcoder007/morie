# -*- coding: utf-8 -*-
"""Tests for targeted learning book chapters 11-29 (tranche B)."""
import importlib
import math

import pytest

tlsate = importlib.import_module("morie.fn.tlsate")
tlsieve = importlib.import_module("morie.fn.tlsieve")
tlstoch = importlib.import_module("morie.fn.tlstoch")
tlclust = importlib.import_module("morie.fn.tlclust")
tlonsl = importlib.import_module("morie.fn.tlonsl")
tlonts = importlib.import_module("morie.fn.tlonts")
tlnetlg = importlib.import_module("morie.fn.tlnetlg")
tlnet1 = importlib.import_module("morie.fn.tlnet1")
tloilr = importlib.import_module("morie.fn.tloilr")
tlbandt = importlib.import_module("morie.fn.tlbandt")
tlcvnp = importlib.import_module("morie.fn.tlcvnp")
tlhoest = importlib.import_module("morie.fn.tlhoest")
tlboot = importlib.import_module("morie.fn.tlboot")
tlsurvy = importlib.import_module("morie.fn.tlsurvy")
np = importlib.import_module("morie.fn._array_core")

RING = {i: [(i - 1) % 30, (i + 1) % 30] for i in range(30)}


def _sate_data(n=6000, seed=2):
    rng = np.random.default_rng(seed)
    A, Y, Q1, Q0, g = [], [], [], [], []
    for _ in range(n):
        w = float(rng.uniform())
        a = 1.0 if float(rng.uniform()) < 0.5 else 0.0
        m1, m0 = 0.2 + 0.6 * w, 0.2 + 0.1 * w
        y = 1.0 if float(rng.uniform()) < (m1 if a == 1.0
                                           else m0) else 0.0
        A.append(a)
        Y.append(y)
        Q1.append(m1)
        Q0.append(m0)
        g.append(0.5)
    return A, Y, Q1, Q0, g


# ------------------------------------------------------------- tlsate
def test_tlsate_variance_gap_is_the_effect_variance():
    A, Y, Q1, Q0, g = _sate_data()
    psi = sum(Q1[i] - Q0[i] for i in range(len(Q1))) / len(Q1)
    vg = tlsate.variance_gap(A, Y, Q1, Q0, g, psi)
    assert abs(vg["gap"] - vg["var_conditional_effect"]) < \
        0.15 * vg["var_conditional_effect"]


def test_tlsate_sample_interval_is_narrower():
    A, Y, Q1, Q0, g = _sate_data(2000)
    fit = tlsate.sate_tmle(A, Y, Q1, Q0, g)
    assert fit["se_sample"] < fit["se_population"]


def test_tlsate_no_effect_modification_no_gap():
    A, Y, Q1, Q0, g = _sate_data(500)
    n = len(A)
    vg = tlsate.variance_gap(A, Y, [0.5] * n, [0.3] * n, g, 0.2)
    assert abs(vg["gap"]) < 1e-9


def test_tlsate_pairs_must_have_two_units():
    with pytest.raises(ValueError):
        tlsate.paired_variance([0, 0, 1], [1.0, 2.0, 3.0])


def test_tlsate_rejects_positivity_violation():
    with pytest.raises(ValueError):
        tlsate.sate_influence_curve([1.0], [1.0], [0.5], [0.5], [0.0])


# ------------------------------------------------------------ tlsieve
def test_tlsieve_incidences_and_survival_close():
    aj = tlsieve.aalen_johansen([1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                                [1, 2, 1, 0, 2, 1],
                                [1.0, 2.0, 3.0])
    assert max(abs(v - 1.0) for v in aj["closure"]) < 1e-12


def test_tlsieve_incidence_is_monotone():
    aj = tlsieve.aalen_johansen([1.0, 2.0, 3.0, 3.0], [1, 2, 1, 2],
                                [1.0, 2.0, 3.0])
    for j in aj["types"]:
        for i in range(1, 3):
            assert aj["F"][j][i] >= aj["F"][j][i - 1] - 1e-12


def test_tlsieve_vaccine_efficacy_and_sieve_effect():
    assert abs(tlsieve.vaccine_efficacy([0.05], [0.10])[0]
               - 0.5) < 1e-12
    se = tlsieve.sieve_effect([0.02], [0.10], [0.08], [0.10])
    assert abs(se["sieve_effect"][0] - 0.6) < 1e-12


def test_tlsieve_no_sieving_is_exactly_zero():
    se = tlsieve.sieve_effect([0.05], [0.10], [0.05], [0.10])
    assert abs(se["sieve_effect"][0]) < 1e-12


def test_tlsieve_rejects_bad_input():
    with pytest.raises(ValueError):
        tlsieve.cause_specific_hazard([1.0], [1, 2], [1.0])
    with pytest.raises(ValueError):
        tlsieve.cumulative_incidence({}, [1.0])


# ------------------------------------------------------------ tlstoch
def _shift_setup(n=400, seed=6):
    rng = np.random.default_rng(seed)
    A = [float(rng.uniform()) * 4.0 for _ in range(n)]
    W = [[float(rng.uniform())] for _ in range(n)]
    return A, W


def test_tlstoch_shift_moves_the_mean_by_the_slope():
    A, W = _shift_setup()
    r = tlstoch.stochastic_estimand(lambda a, w: 0.5 * a + 2.0 * w[0],
                                    A, W, 1.0)
    assert abs(r["contrast"] - 0.5) < 1e-9


def test_tlstoch_large_shift_leaves_the_support():
    A, W = _shift_setup()
    assert tlstoch.positivity_shift(A, 3.5)["fraction_outside"] > \
        tlstoch.positivity_shift(A, 0.2)["fraction_outside"]


def test_tlstoch_clever_covariate_is_a_density_ratio():
    r = tlstoch.density_ratio([1.0, 2.0], [[0.0]] * 2, 0.5,
                              lambda a, w: 0.25)
    assert all(abs(v - 1.0) < 1e-9 for v in r["H"])


def test_tlstoch_truncates_at_the_bounds():
    r = tlstoch.shift_regime([1.0, 5.0], 1.0, lower=0.0, upper=4.0)
    assert r["n_clipped"] == 1
    assert r["shifted"][1] == 4.0


def test_tlstoch_rejects_degenerate_density():
    with pytest.raises(ValueError):
        tlstoch.density_ratio([1.0], [[0.0]], 0.5,
                              lambda a, w: 0.0)


# ------------------------------------------------------------ tlclust
def _clustered(seed=8, J=20, per=10, correlated=True):
    rng = np.random.default_rng(seed)
    ic, cl = [], []
    for j in range(J):
        shared = (float(rng.uniform()) - 0.5) * 2.0 if correlated \
            else 0.0
        for _ in range(per):
            ic.append(shared + 0.2 * (float(rng.uniform()) - 0.5)
                      if correlated
                      else float(rng.uniform()) - 0.5)
            cl.append(j)
    return ic, cl


def test_tlclust_clustering_widens_the_standard_error():
    ic, cl = _clustered()
    assert tlclust.design_effect(ic, cl)["ratio"] > 2.0


def test_tlclust_independent_data_gives_ratio_near_one():
    ic, cl = _clustered(correlated=False)
    assert abs(tlclust.design_effect(ic, cl)["ratio"] - 1.0) < 0.35


def test_tlclust_needs_at_least_two_clusters():
    with pytest.raises(ValueError):
        tlclust.cluster_variance([1.0, 2.0], [0, 0])


def test_tlclust_both_parametrizations_run():
    a = tlclust.g_formula_pooled([0.4, 0.6])
    b = tlclust.g_formula_sequential([[0.4, 0.6], [0.5, 0.5]])
    assert abs(a["psi"] - 0.5) < 1e-12
    assert b["T"] == 2


# ------------------------------------------------------------- tlonsl
def _ar_series(n=220, seed=12):
    rng = np.random.default_rng(seed)
    out, prev = [], 0.0
    for _ in range(n):
        prev = 0.8 * prev + (float(rng.uniform()) - 0.5)
        out.append(prev)
    return out


def _lag_alg(past):
    return lambda z: (0.8 * z[-1]) if z else 0.0


def _mean_alg(past):
    m = sum(past) / len(past)
    return lambda z: m


def test_tlonsl_sequential_validation_prefers_the_lag_model():
    s = _ar_series()
    r = tlonsl.online_super_learner(s, {"lag": _lag_alg,
                                        "mean": _mean_alg},
                                    burn_in=20)
    assert r["best_member"] == "lag"
    assert r["weights"]["lag"] > r["weights"]["mean"]


def test_tlonsl_empty_summary_is_the_iid_case():
    assert tlonsl.summary_measure([1.0, 2.0], lags=0) == []


def test_tlonsl_scores_only_held_out_points():
    s = _ar_series()
    r = tlonsl.sequential_risk(s, _mean_alg, burn_in=20)
    assert r["n_scored"] == len(s) - 20


def test_tlonsl_rejects_bad_input():
    s = _ar_series(50)
    with pytest.raises(ValueError):
        tlonsl.sequential_risk(s, _mean_alg, burn_in=0)
    with pytest.raises(ValueError):
        tlonsl.online_super_learner(s, {})
    with pytest.raises(ValueError):
        tlonsl.sequential_risk(s, _mean_alg, loss="hinge")


# ------------------------------------------------------------- tlonts
def test_tlonts_martingale_check_separates_the_two_cases():
    rng = np.random.default_rng(77)
    past = [float(rng.uniform()) for _ in range(400)]
    D = [(1.0 if i % 2 == 0 else -1.0) * float(rng.uniform())
         for i in range(400)]
    assert tlonts.martingale_check(D, past)["is_martingale"]
    assert not tlonts.martingale_check(past, past)["is_martingale"]


def test_tlonts_intervention_touches_only_its_nodes():
    r = tlonts.stochastic_intervention([1.0, 0.0, 1.0, 0.0], [1, 3],
                                       prob=1.0)
    assert r["intervened"] == [1.0, 1.0, 1.0, 1.0]
    assert r["n_intervened"] == 2


def test_tlonts_variance_is_the_sum_of_squares():
    D = [1.0, -1.0, 0.5]
    r = tlonts.martingale_variance(D)
    assert abs(r["variance"] - sum(v * v for v in D) / 3) < 1e-12


def test_tlonts_rejects_bad_input():
    with pytest.raises(ValueError):
        tlonts.stochastic_intervention([1.0], [0], shift=1.0,
                                       prob=0.5)
    with pytest.raises(ValueError):
        tlonts.stochastic_intervention([1.0], [5], shift=1.0)
    with pytest.raises(ValueError):
        tlonts.martingale_variance([1.0])


# ------------------------------------------------------------ tlnetlg
def test_tlnetlg_hub_violates_the_degree_condition():
    hub = {0: list(range(1, 30))}
    for i in range(1, 30):
        hub[i] = [0]
    assert tlnetlg.network_summary(RING)["sparse"]
    assert not tlnetlg.network_summary(hub)["sparse"]


def test_tlnetlg_connected_pairs_add_covariance():
    rng = np.random.default_rng(21)
    shared = [(float(rng.uniform()) - 0.5) for _ in range(30)]
    ic = [shared[i] + shared[(i + 1) % 30] for i in range(30)]
    r = tlnetlg.network_variance(ic, RING)
    assert r["se"] > r["se_naive"]
    assert r["n_dependent_pairs"] == 60


def test_tlnetlg_exposure_summary_is_a_fraction():
    es = tlnetlg.exposure_summary([1.0, 0.0, 1.0, 0.0],
                                  {0: [1, 2], 1: [0], 2: [0, 3],
                                   3: [2]})["summary"]
    assert abs(es[0][1] - 0.5) < 1e-12


def test_tlnetlg_rejects_bad_input():
    with pytest.raises(ValueError):
        tlnetlg.network_summary({0: []})
    with pytest.raises(ValueError):
        tlnetlg.exposure_summary([1.0, 0.0], {0: [1]})


# ------------------------------------------------------------- tlnet1
def _Qnet(own, frac, w):
    return 0.1 + 0.5 * own + 0.3 * frac + 0.2 * w[0]


def test_tlnet1_decomposes_direct_and_spillover():
    W = [[0.5] for _ in range(30)]
    d = tlnet1.decompose_effects(_Qnet, W, RING, draws=60)
    assert abs(d["direct"] - 0.5) < 0.02
    assert abs(d["spillover"] - 0.3) < 0.02


def test_tlnet1_policy_mean_is_a_policy_property():
    W = [[0.5] for _ in range(30)]
    r = tlnet1.policy_mean(_Qnet, W, RING, 0.0, draws=40)
    assert abs(r["psi"] - 0.2) < 1e-9


def test_tlnet1_flags_an_asymmetric_edge():
    assert not tlnet1.check_network_assumption({0: [1],
                                                1: []})["symmetric"]
    assert tlnet1.check_network_assumption({0: [1],
                                            1: [0]})["symmetric"]


def test_tlnet1_rejects_bad_input():
    W = [[0.5] for _ in range(30)]
    with pytest.raises(ValueError):
        tlnet1.policy_mean(_Qnet, W, RING, 1.5)
    with pytest.raises(ValueError):
        tlnet1.friend_summary([1.0, 0.0], {0: [1]})


# ------------------------------------------------------------- tloilr
Q1V = [0.6, 0.5, 0.4, 0.2, 0.1]
Q0V = [0.3, 0.3, 0.3, 0.3, 0.3]


def test_tloilr_budget_binds_and_costs_value():
    a = tloilr.constrained_value(Q1V, Q0V, 0.4)
    b = tloilr.constrained_value(Q1V, Q0V, 1.0)
    assert a["treated_fraction"] <= 0.4 + 1e-9
    assert a["value"] < b["value"]


def test_tloilr_unbinding_budget_recovers_the_optimal_rule():
    b = tloilr.constrained_value(Q1V, Q0V, 1.0)
    assert abs(b["value"] - b["unconstrained_value"]) < 1e-12
    assert b["tau"] == 0.0


def test_tloilr_flags_an_exceptional_law():
    r = tloilr.exceptional_law([0.2, 0.0, 0.0, -0.1])
    assert r["exceptional"]
    assert abs(r["mass_at_zero"] - 0.5) < 1e-12


def test_tloilr_threshold_is_never_negative():
    assert tloilr.resource_threshold([-0.5, -0.4], 0.5)["tau"] == 0.0


def test_tloilr_rejects_bad_kappa():
    with pytest.raises(ValueError):
        tloilr.resource_threshold([0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        tloilr.blip([0.1], [0.1, 0.2])


# ------------------------------------------------------------ tlbandt
def _bandit(n=400, seed=31):
    rng = np.random.default_rng(seed)
    W = [[float(rng.uniform())] for _ in range(n)]
    Y1 = [0.6 + 0.1 * float(rng.uniform()) for _ in range(n)]
    Y0 = [0.4 + 0.1 * float(rng.uniform()) for _ in range(n)]
    return W, Y1, Y0


def test_tlbandt_bounded_randomisation():
    assert tlbandt.design_probability(0.5, 0.1) == 0.9
    assert tlbandt.design_probability(-0.5, 0.1) == 0.1
    assert tlbandt.design_probability(0.5, 0.1, greedy=True) == 1.0


def test_tlbandt_design_stays_in_the_interval():
    W, Y1, Y0 = _bandit()
    r = tlbandt.run_bandit(W, Y1, Y0, lambda h: 0.2, seed=3)
    assert r["min_g"] >= 0.1 - 1e-12
    assert r["max_g"] <= 0.9 + 1e-12


def test_tlbandt_inference_costs_regret():
    W, Y1, Y0 = _bandit()
    b = tlbandt.run_bandit(W, Y1, Y0, lambda h: 0.2, seed=3)
    g = tlbandt.run_bandit(W, Y1, Y0, lambda h: 0.2, seed=3,
                           greedy=True)
    assert tlbandt.regret(b["Y"], Y1, Y0)["cumulative_regret"] > \
        tlbandt.regret(g["Y"], Y1, Y0)["cumulative_regret"]


def test_tlbandt_greedy_design_is_rejected_for_inference():
    W, Y1, Y0 = _bandit(100)
    g = tlbandt.run_bandit(W, Y1, Y0, lambda h: 0.2, seed=3,
                           greedy=True)
    with pytest.raises(ValueError):
        tlbandt.martingale_terms(g["A"], g["Y"], g["g"],
                                 [0.6] * 100, [0.4] * 100, 0.2)


def test_tlbandt_rejects_bad_delta():
    with pytest.raises(ValueError):
        tlbandt.design_probability(0.1, 0.6)


# ------------------------------------------------------------- tlcvnp
def _uniform(n=2000, seed=41):
    rng = np.random.default_rng(seed)
    return [float(rng.uniform()) for _ in range(n)]


def test_tlcvnp_smoothed_density_recovers_the_truth():
    r = tlcvnp.smoothed_parameter(_uniform(), 0.5, 0.2)
    assert abs(r["psi_h"] - 1.0) < 0.1


def test_tlcvnp_narrow_bandwidth_costs_variance():
    X = _uniform()
    assert tlcvnp.smoothed_parameter(X, 0.5, 0.05)["se"] > \
        tlcvnp.smoothed_parameter(X, 0.5, 0.4)["se"]


def test_tlcvnp_bias_order_is_h_to_the_s():
    a = tlcvnp.smoothing_bias(None, 0.5, 0.2, 2.0)["bias_order"]
    b = tlcvnp.smoothing_bias(None, 0.5, 0.4, 2.0)["bias_order"]
    assert abs(b / a - 4.0) < 1e-9


def test_tlcvnp_bandwidth_selected_from_the_data():
    sel = tlcvnp.select_bandwidth(_uniform(), 0.5,
                                  [0.05, 0.1, 0.2, 0.4])
    assert sel["h"] in (0.05, 0.1, 0.2, 0.4)


def test_tlcvnp_rejects_bad_input():
    with pytest.raises(ValueError):
        tlcvnp.smoothed_parameter([0.1], 0.5, 0.0)
    with pytest.raises(ValueError):
        tlcvnp.kernel_smooth(0.0, "triangular")
    with pytest.raises(ValueError):
        tlcvnp.select_bandwidth([0.1], 0.5, [])


# ------------------------------------------------------------ tlhoest
def test_tlhoest_higher_order_relaxes_the_rate():
    assert abs(tlhoest.rate_requirement(1)
               ["required_rate_per_nuisance"] - 0.25) < 1e-12
    assert abs(tlhoest.rate_requirement(2)
               ["required_rate_per_nuisance"] - 1 / 6.0) < 1e-12


def test_tlhoest_u_statistic_excludes_the_diagonal():
    obs = [1.0, 2.0, 3.0, 4.0]
    a = tlhoest.second_order_term(lambda x, y: x * y, obs)
    b = tlhoest.second_order_term(lambda x, y: x * y, obs,
                                  exclude_diagonal=False)
    assert a["n_pairs"] == 12
    assert abs(a["value"] - b["value"]) > 1e-6


def test_tlhoest_zero_kernel_reduces_to_first_order():
    r = tlhoest.higher_order_estimate(0.5, [0.1, -0.1],
                                      lambda x, y: 0.0, [1.0, 2.0])
    assert abs(r["psi"] - r["first_order"]) < 1e-12


def test_tlhoest_names_the_remainder_order():
    assert tlhoest.remainder_order(1)["remainder_order"] == 2
    assert tlhoest.remainder_order(2)["remainder_order"] == 3


def test_tlhoest_rejects_bad_input():
    with pytest.raises(ValueError):
        tlhoest.second_order_term(lambda x, y: 1.0, [1.0])
    with pytest.raises(ValueError):
        tlhoest.rate_requirement(0)


# ------------------------------------------------------------- tlboot
def test_tlboot_multiplier_matches_the_influence_curve_se():
    rng = np.random.default_rng(51)
    ic = [(float(rng.uniform()) - 0.5) * 2.0 for _ in range(500)]
    r = tlboot.multiplier_bootstrap(ic, B=400, seed=1)
    assert abs(r["ratio"] - 1.0) < 0.12


def test_tlboot_unstable_refitting_inflates_the_spread():
    rng = np.random.default_rng(51)
    data = [0.3 + (float(rng.uniform()) - 0.5) for _ in range(200)]

    def stable(s):
        return sum(s) / len(s)

    def unstable(s):
        t = sorted(s)
        m = sum(t) / len(t)
        skew = sum((v - m) ** 3 for v in t) / len(t)
        return m if skew > 0 else t[len(t) // 2] + 0.05

    a = tlboot.naive_bootstrap(data, stable, B=200, seed=2)
    b = tlboot.naive_bootstrap(data, unstable, B=200, seed=2)
    assert b["se"] > a["se"]


def test_tlboot_targeted_bootstrap_gets_two_moments():
    rng = np.random.default_rng(53)
    data = [0.3 + (float(rng.uniform()) - 0.5) for _ in range(200)]
    naive = tlboot.naive_bootstrap(data, lambda s: sum(s) / len(s),
                                   B=200, seed=2)

    def sampler(r):
        return [0.3 + (float(r.uniform()) - 0.5) for _ in range(200)]

    tb = tlboot.targeted_bootstrap(sampler,
                                   lambda s: sum(s) / len(s),
                                   B=200, seed=3)
    assert tlboot.moment_check(tb["replicates"], 0.3,
                               naive["se"])["first_two_moments_ok"]


def test_tlboot_rejects_bad_input():
    with pytest.raises(ValueError):
        tlboot.naive_bootstrap([1.0], lambda s: 0.0)
    with pytest.raises(ValueError):
        tlboot.multiplier_bootstrap([1.0])


# ------------------------------------------------------------ tlsurvy
def _survey(N=2000):
    vals = [1.0 if i < 100 else 0.01 for i in range(N)]
    return vals, [abs(v) for v in vals], N


def test_tlsurvy_inclusion_probabilities_sum_to_n():
    vals, infl, N = _survey()
    pi = tlsurvy.inclusion_probabilities(vals, 200, "adaptive",
                                         infl)["pi"]
    assert abs(sum(pi) - 200) < 1.0
    assert min(pi) > 0.0


def test_tlsurvy_horvitz_thompson_is_unbiased():
    vals, infl, N = _survey()
    pi = tlsurvy.inclusion_probabilities(vals, 200, "adaptive",
                                         infl)["pi"]
    s = tlsurvy.draw_sample(pi, seed=5)
    ht = tlsurvy.horvitz_thompson(vals, pi, s["selected"])
    truth = sum(vals) / N
    assert abs(ht["estimate"] - truth) < 0.3 * truth


def test_tlsurvy_adaptive_beats_uniform_when_influence_concentrates():
    vals, infl, N = _survey()
    assert tlsurvy.design_efficiency(vals, infl, 200,
                                     seed=7)["ratio"] < 0.9


def test_tlsurvy_flat_influence_buys_nothing():
    flat = [1.0] * 2000
    r = tlsurvy.design_efficiency(flat, flat, 200, seed=7)
    assert abs(r["ratio"] - 1.0) < 0.15


def test_tlsurvy_rejects_bad_input():
    with pytest.raises(ValueError):
        tlsurvy.inclusion_probabilities([1.0, 2.0], 5)
    with pytest.raises(ValueError):
        tlsurvy.inclusion_probabilities([1.0, 2.0], 1, "adaptive")
    with pytest.raises(ValueError):
        tlsurvy.inclusion_probabilities([1.0, 2.0], 1, "poisson")


def test_tl_book2_cheatsheets_are_present():
    for mod in (tlsate, tlsieve, tlstoch, tlclust, tlonsl, tlonts,
                tlnetlg, tlnet1, tloilr, tlbandt, tlcvnp, tlhoest,
                tlboot, tlsurvy):
        assert len(mod.cheatsheet()) > 80
