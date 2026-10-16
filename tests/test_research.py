# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.research: every function must reproduce the numerical consequences of its Lean theorem and agree
with the R arm (rmorie) to rounding. The parity values below were computed in R on the same inputs."""

import math

import pytest

from morie import research as R

TOL = 1e-12


def close(a, b, tol=TOL):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


# ----------------------------------------------------------------- P1 dark figure


def test_lincoln_petersen_and_dependence_box():
    N, p1, p2 = 1250, 0.32, 0.2
    out = R.dark_figure_two_source(N * p1, N * p2, N * p1 * p2)
    assert (
        close(out["point"], N) and close(out["lower"], N) and out["theorem"] == "Research.P1.TwoSource.lincoln_petersen"
    )
    kappa = 2
    for theta in (1 / kappa, 0.8, 1, 1.5, kappa):
        o = R.dark_figure_two_source(N * p1, N * p2, N * p1 * p2 * theta, kappa=kappa)
        assert o["lower"] <= N + 1e-9 and o["upper"] >= N - 1e-9 and close(o["point"] * theta, N)  # petersen_identity
    assert close(R.dark_figure_two_source(N * p1, N * p2, N * p1 * p2 / kappa, kappa=kappa)["lower"], N)
    assert close(R.dark_figure_two_source(N * p1, N * p2, N * p1 * p2 * kappa, kappa=kappa)["upper"], N)
    with pytest.raises(ValueError, match="smaller list"):
        R.dark_figure_two_source(10, 20, 15)


def test_dark_figure_bounds_attained_and_breakdown():
    b = R.dark_figure_bounds(0.06, 0.02, alpha_max=0.01, beta_max=0.30)
    assert close(b["v_upper"][0], 0.0857142857142857)  # R
    assert close(b["v_lower"][0], max(0.02, (0.06 - 0.01) / 0.99))
    # the survey part is attained: v with beta = beta_max reproduces v_obs
    v = float(b["v_upper"][0])
    assert close(v * (1 - 0.30), 0.06)
    br = R.dark_figure_breakdown(0.06, threshold=0.10)
    assert close(br["breakdown_beta_max"], 0.4) and close(br["ratio"], 0.1 / 0.06)
    with pytest.raises(ValueError, match="below 1"):
        R.dark_figure_bounds(0.1, 0.0, 0.6, 0.5)


def test_three_list_and_hierarchy():
    r = R.dark_figure_three_list(
        {"100": 120, "010": 90, "001": 70, "110": 40, "101": 30, "011": 25, "111": 15}, candidate_missing=[100, 378]
    )
    assert close(r["observed"], 390) and close(r["missing_no_three_way"], 378)  # R
    assert close(r["implied_three_way"]["three_way"][1], 0.0, 1e-9)  # the no-three-way candidate
    for _, row in r["pairwise"].iterrows():
        assert row["petersen"] >= row["floor"] - 1e-9 and row["chapman"] >= row["floor"] - 1e-9  # *_ge_floor
    h = R.dark_figure_hierarchy([1, 1, 2, 1, 3, 1, 1, 2])
    assert (h["incidents"], h["offences"], h["mean_extra"]) == (8, 12, 0.5)
    assert h["bounds"] == {"lower": 8, "upper": 24} and close(h["offences"], 8 * (1 + 0.5))  # offence_count_eq


# ----------------------------------------------------------------- P2, P6, P8


def test_selection_family_parity():
    d = R.disparity_exposure_bounds({"A": 300, "B": 100}, {"A": 1000, "B": 1000}, gamma=1.5, reference="B")
    assert [round(float(v), 12) for v in d["ratio_lower"]] == [1.333333333333, 0.444444444444]
    assert list(d["direction_identified"]) == [True, False]
    bm = R.disparity_benchmark(
        {"A": 100, "B": 100},
        {"A": 30, "B": 10},
        {"A": 12, "B": 2},
        reference="B",
        exposure_error_factor={"A": 2.0, "B": 1.0},
    )
    assert all(close(v, 0) for v in bm["product_check"])  # benchmark_product
    assert close(bm["resident_disparity"][0], 6) and close(bm["additive_claim"][0], 3 + 2)  # not additive
    assert close(bm["log_shift"][0], -math.log(2)) and close(bm["resident_disparity_corrected"][0], 3)
    rr = R.relative_risk_from_or(3, base_rate=0.2, exposed_share=0.3)
    assert close(rr["risks"]["relative_risk"], 2.33333333333333, 1e-10)  # R
    a, b = rr["risks"]["exposed"], rr["risks"]["unexposed"]
    assert close(3, (a / b) * (1 - b) / (1 - a), 1e-9)  # or_eq_rr_mul
    assert 1 <= rr["risks"]["relative_risk"] <= 3  # rr_between
    det = R.deterrence_response([0, 1, 2, 3], [0, 2, 3, 3.5], [0, 1, 3, 6], p=[0.1, 0.5, 1])
    assert list(det["x_opt"]) == [3.0, 2.0, 1.0]  # R; certainty_monotone
    chk = R.deterrence_design_check(p=[0.1, 0.3, 0.5], s=[2, 2, 2], c=[30, 30, 30])
    assert chk["rank"] == 2 and chk["identified"] == {"certainty": True, "severity": False, "celerity": False}
    ir = R.interracial_rates({"A_on_B": 120, "B_on_A": 200, "A_on_A": 900, "B_on_B": 300}, {"A": 80000, "B": 20000})
    assert [round(float(v), 12) for v in ir["rate_per_pair_exposure"]] == [0.0075, 0.0125, 0.0140625, 0.075]
    pn = R.probability_of_necessity(0.6, 0.4)
    assert close(pn["necessary_share_bounds"]["lower"], 0.2) and close(pn["necessary_share_bounds"]["upper"], 0.6)
    assert close(pn["pn_monotone"], 1 / 3)
    col = R.collider_arrest(0.3, 0.2, 0.1)
    assert close(col["arrestee_or"], 0.1) and col["population_or"] == 1  # collider_or_eq_background
    agg = R.age_crime_aggregate([0.5, 0.5], [[1, 3], [2, 2], [3, 1]], ages=[10, 11, 12])
    assert list(agg["aggregate"]) == [2.0, 2.0, 2.0]


# ----------------------------------------------------------------- P3 spillover + Cheeger


def test_spillover_exposure_adjustment_and_ht():
    edges = [[1, 2], [2, 3], [3, 4], [4, 5]]
    e = R.spillover_exposure([1, 0, 0, 0, 1], edges)
    assert list(e["exposure"]) == [2, 1, 0, 1, 2] and list(e["treated_neighbours"]) == [0, 1, 0, 1, 0]
    sp = R.spillover_effects([5, 4.6, 4, 6, 5.6, 5], [0, 1, 2, 0, 1, 2], ["a", "a", "a", "b", "b", "b"])
    assert (
        close(sp["spillover"], -0.4)
        and close(sp["direct"], -0.6)
        and close(sp["total"], sp["spillover"] + sp["direct"])
    )
    edges6 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    pr = R.spillover_exposure_probs(6, edges6, 2, joint=True)
    assert all(close(float(v), 1.0) for v in pr["marginal"].sum(axis=1))  # rows are distributions
    ex = list(R.spillover_exposure([0, 1, 0, 0, 0, 1], edges6)["exposure"])
    ht = R.spillover_ht([5, 3.5, 4.5, 5, 4.5, 3.5], ex, pr["marginal"], pr["joint"])
    assert close(ht["total_effect"], -0.666666666666667, 1e-12)  # R
    assert close(ht["se"]["T0"], 22.3606797749979, 1e-9) and close(ht["se"]["T1"], 23.4905267488572, 1e-9)  # R
    v = R.spillover_ht_variance([1] * 6, pr["marginal"][:, 2], pr["joint"][:, :, 2], form="both")
    assert v["fixed_size"] is True and close(v["level_count"], 2)
    assert close(v["ht"], v["syg"], 1e-9)  # syg_eq_ht on a fixed-size design
    p4 = R.spillover_exposure_probs(4, [[1, 2], [2, 3], [3, 4]], 1)
    assert [round(float(x), 12) for x in p4[:, 2]] == [0.25] * 4


def test_cheeger_identities_and_bound():
    A = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    r = R.cheeger_bound(A, S=[0, 1, 2], exhaustive=True)
    assert close(r["conductance"], 1 / 7) and close(r["rayleigh_test"], 2 / 7)
    assert close(r["lambda2"], 0.204666354556872, 1e-9)  # R
    assert r["bound_holds"] and r["argmin_set"] == [0, 1, 2] and r["cheeger_bound_holds"]
    # the identities on a random weighted graph
    from morie.fn._rng import random_uniform

    n = 7
    u = [float(x) for x in random_uniform(n * n, seed=3)]
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = u[i * n + j]
            M[i][j] = M[j][i] = w if w > 0.3 else 0.0
    S = [0, 2, 5]
    r = R.cheeger_bound(M, S)
    d = [sum(row) for row in M]
    vol_s = sum(d[i] for i in S)
    vol_c = sum(d[i] for i in range(n) if i not in S)
    cut = sum(M[i][j] for i in S for j in range(n) if j not in S)
    f = [1 / vol_s if i in S else -1 / vol_c for i in range(n)]
    assert close(sum(d[i] * f[i] for i in range(n)), 0, 1e-12)  # testVec_orth
    E = 0.5 * sum(M[i][j] * (f[i] - f[j]) ** 2 for i in range(n) for j in range(n))
    assert close(E, cut * (1 / vol_s + 1 / vol_c) ** 2)  # testVec_dirichlet
    assert close(r["rayleigh_test"], cut * (1 / vol_s + 1 / vol_c))  # rayleigh_testVec
    assert r["lambda2"] <= r["rayleigh_test"] + 1e-10 <= 2 * r["conductance"] + 2e-10  # cheeger_easy


# ----------------------------------------------------------------- P4 feedback loop


def test_feedback_loop_theorems_and_seeded_parity():
    mf = R.feedback_loop_meanfield(0.3, 0.2, 7, 3, n_steps=1)
    x = 0.7
    drift = x * (1 - x) * 0.1 / (10 + 0.3 * x + 0.2 * (1 - x))
    assert close(float(mf["share_a"][1] - mf["share_a"][0]), drift)  # naive_step_drift
    mf = R.feedback_loop_meanfield(0.21, 0.20, 1, 99, n_steps=500)
    s = [float(v) for v in mf["share_a"]]
    assert all(b > a for a, b in zip(s, s[1:]))  # naive_share_increasing
    cf = R.feedback_loop_meanfield(0.3, 0.2, 1, 99, n_steps=2000, update="corrected")
    assert close(float(cf["share_a"][2000]), (1 + 2000 * 0.3) / (100 + 2000 * 0.5))  # closed form
    assert R.feedback_loop_limit(0.3, 0.2, 10, 10)["share_a"] == 1
    assert close(R.feedback_loop_limit(0.3, 0.2, 10, 10, rho=0.5)["cap"], 0.75)
    eq = R.feedback_loop_meanfield(0.25, 0.25, 30, 70, n_steps=50)
    assert all(close(v, 0.3) for v in eq["share_a"])
    b = R.feedback_loop_bound(0.3, 0.2, 10, 10, n_steps=4)
    assert close(b["bound"], 0.00492670619146438)  # R
    law = R.feedback_loop_urn_law(10)
    assert (
        close(float(law["prob"][0]), 1 / 11)
        and close(law.attrs["prob_middle"], 7 / 11)
        and law.attrs["prob_middle"] >= 0.25
    )
    sim = R.feedback_loop_sim(0.3, 0.2, 10, 10, n_steps=50, n_sims=3, seed=1)
    assert [round(float(v), 12) for v in sim["final"]] == [
        0.628571428571,
        0.517241379310,
        0.529411764706,
    ]  # R, same Philox stream
    # rho > 0: the proved cap is never exceeded along the mean-field path
    mr = R.feedback_loop_meanfield(0.3, 0.2, 10, 10, n_steps=5000, rho=0.5)
    assert max(float(v) for v in mr["share_a"]) <= 0.75 + 1e-12  # rho_cap


# ----------------------------------------------------------------- P5 fairness, separation


def test_fairness_bounds_family():
    fr = R.fairness_rates(120, 60, 40, 280)
    assert close(R.fairness_implied_fpr(fr["p"], fr["ppv"], fr["fnr"]), fr["fpr"])  # chouldechova
    assert not close(R.fairness_implied_fpr(0.30, 0.6, 0.25), R.fairness_implied_fpr(0.50, 0.6, 0.25))  # impossibility
    bb = R.fairness_base_rate_bounds([0.35, 0.55], alpha_max=0.10, beta_max=0.20)
    assert [round(float(v), 12) for v in bb["upper"]] == [0.4375, 0.6875]
    assert close(float(R.fairness_true_rate(0.4 * 0.8 + 0.6 * 0.1, alpha=0.1, beta=0.2)[0]), 0.4)
    assert R.fairness_compare_groups(0.35, 0.55, alpha_max=0.05, beta_max=0.20)["order"] == "a < b"
    assert R.fairness_compare_groups(0.35, 0.55, alpha_max=0.05, beta_max=0.40)["order"] == "undecided"
    lr = R.logit_rescale(beta=0.8, omitted_var=1)
    assert close(lr["rescale"], 0.87572404422235, 1e-12) and lr["rescale"] < 1  # rescale_lt_one
    assert close(R.logit_rescale(0.8, 0.0)["rescale"], 1)  # rescale_eq_one_iff
    rk = R.ranking_resolution([0.2, 0.35, 0.8], half_width=[0.1, 0.1, 0.05])
    assert rk["n_pairs"] == 3 and close(rk["share_unidentified"], 1 / 3) and close(rk["resolution"], 0.2)
    hz = R.hazard_selection(0.5, 0.5, 0.1, (0.5, 0.8), (0.9, 0.9))
    assert close(hz["period2_hazard_ratio"], 1.18685121107266, 1e-12) and hz["period2_hazard_ratio"] > 1  # hr2_gt_one


def test_logit_separation_lp_and_glm():
    x = [[1, 0.2], [1, -1], [1, 0.5], [0, 0.1], [0, -0.4], [0, 1.2], [0, 0.3]]
    y = [1, 1, 1, 0, 0, 0, 0]
    r = R.logit_separation(y, x)
    assert r["separation"] == "complete" and r["method"] == "lp" and all(m > 0 for m in r["margins"])
    ll = [r["loglik_along"][k] for k in ("t=0", "t=1", "t=10", "t=100")]
    assert all(b > a for a, b in zip(ll, ll[1:])) and all(v < 0 for v in ll) and ll[-1] > -1e-6  # shift, neg, tendsto
    assert close(ll[0], -4.85203026391962, 1e-9)  # R (t=0: b = 0)
    assert R.logit_separation(y, x, method="glm")["separation"] in ("complete", "quasi-complete")
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(600, seed=4)]
    xx = [[u[i] - 0.5, u[200 + i] - 0.5] for i in range(200)]
    yy = [1 if u[400 + i] < 1 / (1 + math.exp(-(2 * xx[i][0] - xx[i][1]))) else 0 for i in range(200)]
    assert R.logit_separation(yy, xx)["separation"] == "none"


# ----------------------------------------------------------------- P7 concentration


def test_concentration_identities():
    x = [0, 0, 0, 1, 9]
    assert close(R.concentration_gini(x), 0.76)
    d = R.concentration_decompose(x)
    assert (
        close(d["zero_share"], 0.6) and close(d["gini_positive"], 0.4) and close(d["identity_check"], 0)
    )  # gini_zero_decomposition
    assert close(d["null_zero_share"], math.exp(-2))  # poisson_zero_prob
    dd = R.concentration_dispersion([0, 0, 1, 3, 0, 2])
    assert (
        close(dd["dispersion_index"], 1.6) and dd["zero_share"] >= dd["null_zero_share"] - 1e-12
    )  # mixture_zero_ge_exp_neg_mean
    g = R.concentration_distinct_growth([1, 2, 1, 3, 2, 1, 4, 1])
    assert close(g["M_hat"], 2.50062374748314, 1e-9) and close(g["expected"][-1], 4, 1e-9)  # R; root recovers K
    assert all(
        lo - 1e-9 <= e <= up + 1e-9 for lo, e, up in zip(g["lower"], g["expected"], g["upper"])
    )  # expectedDistinct_bounds


# ----------------------------------------------------------------- P9, P10, P12


def test_recording_contagion_ecological():
    rm = R.recording_map([[0.7, 0], [0.3, 1]], [100, 400], names=["robbery", "theft"])
    assert rm["recorded"] == {"robbery": 70.0, "theft": 430.0} and rm["regime"] == "reclassification"
    assert close(rm["recorded_total"], rm["true_total"])  # total_invariant_of_colStochastic
    cuff = R.recording_map([[0.7, 0], [0.2, 1]], [100, 400])
    assert cuff["regime"] == "cuffing" and close(cuff["true_total"] - cuff["recorded_total"], cuff["dropped"]["total"])
    ds = R.detection_rate_shift(2000, 10000, 1500, 0.2, 0.5)
    assert close(ds["rise"], 0.06) and close(ds["rate_after"] - ds["rate_before"], ds["rise"])  # detection_rate_rises
    c = R.contagion_branching(0.4, mu=2)
    assert (
        close(c["expected_cluster_size"], 1 / 0.6)
        and close(c["stationary_rate"], 2 / 0.6)
        and c["endogeneity_share"] == 0.4
    )
    assert close(sum(c["generation_means"]), sum(0.4**k for k in range(10)))
    assert R.contagion_branching(1.1, 2)["expected_cluster_size"] == float("inf")  # cluster_size_diverges_of_ge_one
    ec = R.ecological_decompose([0, 2, 1, 3], [1, 3, 0, 2], ["a", "a", "b", "b"])
    assert (
        close(ec["corr_individual"], 0.6) and close(ec["corr_ecological"], -1) and ec["sign_reversed"]
    )  # robinson_reversal
    assert close(ec["cov"]["individual"], ec["cov"]["between"] + ec["cov"]["within"])  # cov_decomp
    assert close(ec["var_x"]["individual"], ec["var_x"]["between"] + ec["var_x"]["within"])  # var_decomp


# ----------------------------------------------------------------- P11 sentencing bounds + coverage


def test_sentence_bounds_and_imbens_manski():
    y = [1, 0, 1, 0, 1, 1]
    z = ["a", "a", "a", "b", "b", "b"]
    b = R.sentence_effect_bounds(y, z)
    assert (
        close(b["ate_width"], 1) and b["ate_bounds"]["lower"] <= 0 <= b["ate_bounds"]["upper"]
    )  # width one, contains zero
    assert b["ate_bounds"]["lower"] <= b["naive_difference"] <= b["ate_bounds"]["upper"]
    m = R.sentence_effect_mtr(y, z)
    assert m["bounds"]["lower"] == 0 and close(m["bounds"]["upper"], 0.5)  # mtr_lower / mtr_upper
    cb = R.contaminated_bounds([0.05, 0.5, 0.97], 0.1)
    assert [round(float(v), 12) for v in cb["lower"]] == [0.0, 0.444444444444, 0.966666666667]
    assert all(close(float(w), 0.1 / 0.9) for w in cb["width"])  # clean_width
    im = R.bounds_confidence(0.10, 0.35, 0.03, 0.04)
    assert close(im["cutoff"], 1.64485362695149, 1e-9)  # R
    from morie.fn._stats_core import norm

    assert close(float(norm.cdf(im["cutoff"] + im["delta"])) - float(norm.cdf(-im["cutoff"])), 0.95, 1e-9)
    assert im["z_one_sided"] - 1e-9 <= im["cutoff"] <= im["z_two_sided"] + 1e-9  # im_cutoff_between
    assert im["coverage_two_sided"] > 0.95  # two_sided_overcovers
    cuts = [R.bounds_confidence(0, d * 0.1, 0.1, 0.1)["cutoff"] for d in (0, 0.5, 1, 2, 8)]
    assert all(b2 <= a2 + 1e-9 for a2, b2 in zip(cuts, cuts[1:]))  # im_cutoff_antitone
    assert close(cuts[0], float(norm.ppf(0.975)), 1e-8)


# ----------------------------------------------------------------- P13 pooling


def test_meta_pooling_theorems_and_seeded_parity():
    est = [-0.25, -0.10, -0.40, 0.05, -0.30]
    v = [0.010, 0.020, 0.015, 0.030, 0.012]
    m = R.meta_random_effects(est, v)
    assert close(m["tau2"], 0.00696774193548387, 1e-12) and close(m["variance_ratio"], 1.50618497416259, 1e-12)  # R
    assert m["random"]["variance"] >= m["fixed"]["variance"] and not m["truncated"]  # re_var_ge
    w = [1 / x for x in v]
    theta = sum(a * b for a, b in zip(w, est)) / sum(w)
    q = sum(a * (b - theta) ** 2 for a, b in zip(w, est))
    c = sum(w) - sum(a * a for a in w) / sum(w)
    assert close(m["Q"], q) and close(m["tau2"], max(0, (q - 4) / c))
    hom = R.meta_random_effects([0.1, 0.1, 0.1], [0.01, 0.02, 0.03])
    assert hom["tau2"] == 0 and hom["truncated"] and close(hom["variance_ratio"], 1)  # re_var_eq_iff
    b = R.meta_dl_bias(v, n_draws=500, seed=1)
    assert close(b["mean_tau2"], 0.00368358001446555, 1e-12) and close(b["share_positive"], 0.364)  # R, same stream
    assert close(b["mean_Q"], 3.78725142175425, 1e-12) and b["mean_tau2"] > 0  # pos_part_pos


def test_top_level_lazy_exports():
    import morie

    for name in R.__all__:
        assert getattr(morie, name) is getattr(R, name)


def test_duncan_davis_bounds():
    r = R.ecological_bounds([0.2, 0.5, 0.8], [0.1, 0.3, 0.6], weights=[1000, 2000, 500])
    nb = r["neighbourhoods"]
    assert [round(float(v), 12) for v in nb["upper"]] == [0.5, 0.6, 0.75]
    assert [round(float(v), 12) for v in nb["lower"]] == [0.0, 0.0, 0.5]
    m = [1000 * 0.2, 2000 * 0.5, 500 * 0.8]
    assert close(
        r["aggregate"]["upper"], sum(a * b for a, b in zip(m, [0.5, 0.6, 0.75])) / sum(m)
    )  # dd_aggregate_bounds
    # every admissible joint mass gives a rate inside the interval, and the complement identity holds
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(300, seed=9)]
    for k in range(100):
        p = 0.05 + 0.9 * u[k]
        q = u[100 + k]
        b = R.ecological_bounds(p, q)["neighbourhoods"]
        lo, hi = max(0.0, p + q - 1), min(p, q)
        pq = lo + (hi - lo) * u[200 + k]
        rr = pq / p
        rc = (q - pq) / (1 - p)
        assert float(b["lower"][0]) - 1e-12 <= rr <= float(b["upper"][0]) + 1e-12  # dd_bounds
        assert float(b["complement_lower"][0]) - 1e-12 <= rc <= float(b["complement_upper"][0]) + 1e-12
        assert close(q, p * rr + (1 - p) * rc)  # dd_complement
        assert close(float(b["lower"][0]), lo / p) and close(float(b["upper"][0]), hi / p)  # ends_attained
    with pytest.raises(ValueError, match="equal length"):
        R.ecological_bounds([0.2, 0.5], [0.1])
    with pytest.raises(ValueError, match="p must"):
        R.ecological_bounds([1.0], [0.5])
    with pytest.raises(ValueError, match="positive"):
        R.ecological_bounds([0.5], [0.5], weights=[0])


def test_monotone_treatment_selection():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(2000, seed=11)]
    z = ["b" if u[i] < 0.5 else "a" for i in range(400)]
    y = [1 if u[400 + i] < (0.6 if z[i] == "b" else 0.3) else 0 for i in range(400)]
    r = R.sentence_effect_mts(y, z)
    nb = [y[i] for i in range(400) if z[i] == "b"]
    na = [y[i] for i in range(400) if z[i] == "a"]
    m_b = sum(nb) / len(nb)
    m_a = sum(na) / len(na)
    pzb = len(nb) / 400
    pza = 1 - pzb
    assert close(r["naive_difference"], m_b - m_a) and close(
        r["ate_bounds_mts"]["upper"], m_b - m_a
    )  # mts_ate_le_naive
    assert close(r["mean_b_bounds"]["lower"], pzb * m_b) and close(r["mean_b_bounds"]["upper"], m_b)  # mts_mean_b_le
    assert close(r["mean_a_bounds"]["lower"], m_a) and close(r["mean_a_bounds"]["upper"], pza * m_a + pzb)
    assert r["ate_bounds_mtr_mts"]["lower"] == 0 and close(
        r["ate_bounds_mtr_mts"]["upper"], max(0.0, m_b - m_a)
    )  # mtr_mts_bounds
    # MTS-consistent completions stay inside
    k = 0
    for rep in range(20):
        pb = u[800 + rep] * m_b
        pa = m_a + (1 - m_a) * u[820 + rep]
        yb = list(y)
        ya = list(y)
        for i in range(400):
            if z[i] == "a":
                yb[i] = 1 if u[1000 + i] < pb else 0
            else:
                ya[i] = 1 if u[1400 + i] < pa else 0
        mb_a = sum(ya[i] for i in range(400) if z[i] == "b") / len(nb)
        ma_a = sum(ya[i] for i in range(400) if z[i] == "a") / len(na)
        mb_b = sum(yb[i] for i in range(400) if z[i] == "b") / len(nb)
        ma_b = sum(yb[i] for i in range(400) if z[i] == "a") / len(na)
        if mb_a >= ma_a and mb_b >= ma_b:
            k += 1
            ate = sum(yb) / 400 - sum(ya) / 400
            assert r["ate_bounds_mts"]["lower"] - 1e-12 <= ate <= r["ate_bounds_mts"]["upper"] + 1e-12
    assert k > 0


def test_contagion_extinction_fixed_point():
    r = R.contagion_extinction([0.3, 0.3, 0.4])
    assert close(r["extinction"], 0.75, 1e-9) and r["regime"] == "supercritical"  # 0.4 s^2 - 0.7 s + 0.3 = 0
    it = r["iterates"]
    assert all(b >= a - 1e-15 for a, b in zip(it, it[1:]))  # iter_mono
    assert abs(r["fixed_point_check"]) < 1e-12  # extinction_fixed
    assert close(R.contagion_extinction([0.5, 0.3, 0.2])["extinction"], 1, 1e-9)  # subcritical_extinction_one
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(400, seed=13)]
    for k in range(40):
        K = 2 + (k % 5)
        w = [u[k * 8 + j] for j in range(K + 1)]
        p = [v / sum(w) for v in w]
        rr = R.contagion_extinction(p)
        m = sum(j * p[j] for j in range(K + 1))
        assert close(rr["mean_offspring"], m)
        if m < 1:
            assert close(rr["extinction"], 1, 1e-9)
        if m > 1:
            assert rr["extinction"] < 1 - 1e-9  # supercritical_extinction_lt_one
        f = lambda s, p=p: sum(p[j] * s**j for j in range(len(p)))  # noqa: E731
        if rr["extinction"] > 1e-6:
            grid = [rr["extinction"] * 0.999 * t / 199 for t in range(200)]
            assert all(f(s) > s for s in grid)  # extinction_le_fixed: smallest
    with pytest.raises(ValueError, match="summing to one"):
        R.contagion_extinction([0.5, 0.6])


def test_judge_leniency_instrument():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(2000, seed=14)]
    n = 300
    d0 = [1 if u[i] < 0.3 else 0 for i in range(n)]
    d1 = [1 if u[300 + i] < 0.6 else 0 for i in range(n)]
    y0 = [u[600 + i] for i in range(n)]
    y1 = [y0[i] + u[900 + i] * 2 for i in range(n)]
    w = [u[1200 + i] for i in range(n)]
    r = R.judge_iv_population(d0, d1, y0, y1, w)
    ww = [x / sum(w) for x in w]
    comp = [i for i in range(n) if d0[i] == 0 and d1[i] == 1]
    de = [i for i in range(n) if d0[i] == 1 and d1[i] == 0]
    assert close(
        r["itt"], sum(ww[i] * (y1[i] - y0[i]) for i in comp) - sum(ww[i] * (y1[i] - y0[i]) for i in de)
    )  # itt_decomposition
    assert close(r["first_stage"], sum(ww[i] for i in comp) - sum(ww[i] for i in de))  # first_stage
    pc = r["shares"]["complier"]
    pdf = r["shares"]["defier"]
    assert close(
        r["wald"], (pc * r["effects"]["complier"] - pdf * r["effects"]["defier"]) / (pc - pdf), 1e-9
    )  # wald_with_defiers
    d1m = [max(a, b) for a, b in zip(d0, d1)]
    rm_ = R.judge_iv_population(d0, d1m, y0, y1, w)
    assert rm_["monotone"] and close(rm_["wald"], rm_["late"], 1e-9)  # late_identification
    wit = R.judge_iv_population([0, 1], [1, 0], [0, 0], [1, 4], weights=[0.6, 0.4])
    assert (
        close(wit["wald"], -5) and wit["effects"]["complier"] > 0 and wit["effects"]["defier"] > 0
    )  # defiers_can_flip
    # cloned design: observed means are the population means, so the Wald ratio is the LATE exactly
    z = [0] * n + [1] * n
    d = d0 + d1m
    y = [y1[i] if d0[i] == 1 else y0[i] for i in range(n)] + [y1[i] if d1m[i] == 1 else y0[i] for i in range(n)]
    obs = R.judge_iv(z, d, y, weights=w + w)
    assert close(obs["wald"], rm_["late"]) and close(obs["first_stage"], rm_["first_stage"])
    assert close(obs["shares_if_monotone"]["complier"], rm_["shares"]["complier"])
    s = obs["sensitivity"]
    assert close(float(s["implied_complier_effect"][0]), obs["wald"])
    row = {k: float(s[k][2]) for k in s.columns}
    assert close(
        obs["wald"],
        (row["complier_share"] * row["implied_complier_effect"] - 0.1 * row["defier_effect"])
        / (row["complier_share"] - 0.1),
    )
    with pytest.raises(ValueError, match="both values"):
        R.judge_iv([1, 1], [0, 1], [1, 2])
    with pytest.raises(ValueError, match="0/1"):
        R.judge_iv_population([0, 2], [1, 1], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="equal length"):
        R.judge_iv_population([0], [1, 1], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="equal length"):
        R.judge_iv([0, 1], [0], [1, 2])
    with pytest.raises(ValueError, match="weights"):
        R.judge_iv([0, 1], [0, 1], [1, 2], weights=[1])
    with pytest.raises(ValueError, match="weights"):
        R.judge_iv_population([0, 1], [1, 1], [0, 0], [1, 1], weights=[-1, 1])
    with pytest.raises(ValueError, match="0/1"):
        R.judge_iv([0, 1], [0, 2], [1, 2])
    assert (
        R.judge_iv_population([1, 1], [1, 1], [0, 0], [1, 1])["wald"]
        != R.judge_iv_population([1, 1], [1, 1], [0, 0], [1, 1])["wald"]
    )  # nan
    assert (
        R.judge_iv([0, 0, 1, 1], [1, 1, 1, 1], [1, 2, 3, 4])["wald"]
        != R.judge_iv([0, 0, 1, 1], [1, 1, 1, 1], [1, 2, 3, 4])["wald"]
    )


def test_oaxaca_blinder_identities_and_attribution_shift():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(1200, seed=16)]
    n = 300
    g = ["A"] * 150 + ["B"] * 150
    score = [u[i] * 4 + (5 if g[i] == "A" else 4) for i in range(n)]
    priors = [int(u[300 + i] * (4 if g[i] == "A" else 2)) for i in range(n)]
    y = [2 + 0.8 * score[i] + 0.5 * priors[i] + (1 if g[i] == "A" else 0) + u[600 + i] - 0.5 for i in range(n)]
    X = [[score[i], priors[i]] for i in range(n)]
    r = R.disparity_decomposition(y, X, g, reference="B", names=["score", "priors"], shift=3)
    assert all(abs(v) < 1e-10 for v in r["identity_checks"].values())  # twofold_B/A, threefold, reference
    assert close(r["gap"], sum(y[:150]) / 150 - sum(y[150:]) / 150)
    assert close(r["twofold_A"]["explained"] - r["twofold_B"]["explained"], r["threefold"]["interaction"], 1e-10)
    X2 = [[score[i] + 3, priors[i]] for i in range(n)]
    r2 = R.disparity_decomposition(y, X2, g, reference="B", names=["score", "priors"])
    assert close(
        r2["twofold_B"]["unexplained"], r["twofold_B"]["unexplained"], 1e-9
    )  # attribution_shift: total unchanged
    bv = r["by_variable"]
    bv2 = r2["by_variable"]
    assert close(
        float(bv2["unexplained_A"][1]) - float(bv["unexplained_A"][1]), float(bv["unexplained_shift"][1]), 1e-9
    )
    assert close(float(sum(bv["unexplained_A"])), r["twofold_B"]["unexplained"], 1e-9)
    # identical coefficients: interaction zero and the references agree (explained_eq_iff)
    x = [u[900 + i] + (1 if g[i] == "A" else 0) for i in range(n)]
    y0 = [1 + 2 * v for v in x]
    r0 = R.disparity_decomposition(y0, x, g, reference="B")
    assert abs(r0["threefold"]["interaction"]) < 1e-9 and close(
        r0["twofold_A"]["explained"], r0["twofold_B"]["explained"], 1e-9
    )
    with pytest.raises(ValueError, match="two values"):
        R.disparity_decomposition(y0, x, ["A"] * n, reference="A")
    with pytest.raises(ValueError, match="reference"):
        R.disparity_decomposition(y0, x, g, reference="C")
    with pytest.raises(ValueError, match="collinear"):
        R.disparity_decomposition(y0, [[v, 2 * v] for v in x], g, reference="B")
    with pytest.raises(ValueError, match="same rows"):
        R.disparity_decomposition(y0[:10], x, g, reference="B")


def test_oaxaca_blinder_parity_with_r():
    y = [10, 12, 13, 15, 9, 8, 11, 7]
    X = [[1, 0], [2, 1], [3, 1], [4, 0], [1, 1], [2, 0], [2, 0], [3, 1]]
    r = R.disparity_decomposition(y, X, ["A"] * 4 + ["B"] * 4, reference="B", names=["x1", "x2"], shift=2)
    assert close(r["gap"], 3.75)
    assert close(r["twofold_B"]["explained"], -0.5, 1e-9) and close(r["twofold_B"]["unexplained"], 4.25, 1e-9)  # R
    assert close(r["twofold_A"]["explained"], 0.8, 1e-9) and close(r["twofold_A"]["unexplained"], 2.95, 1e-9)  # R
    assert close(r["threefold"]["interaction"], 1.3, 1e-9)
    bv = r["by_variable"]
    assert [round(float(v), 9) for v in bv["unexplained_A"]] == [-3.0, 6.5, 0.75]  # R
    assert [round(float(v), 9) for v in bv["unexplained_shift"]] == [-6.0, 5.2, 3.0]  # R


def test_little_law_on_a_docket():
    from morie.fn._rng import random_uniform
    from morie.research.court_backlog import occupancy_on_grid

    u = [float(v) for v in random_uniform(400, seed=17)]
    for k in range(10):
        n = 5 + k * 5
        a = [u[i] * 80 for i in range(n)]
        d = [a[i] + u[100 + i] * 20 for i in range(n)]
        r = R.court_backlog(a, d, horizon=100, target_backlog=2.5)
        assert close(r["occupancy_integral"], sum(dv - av for av, dv in zip(a, d)))  # occupancy_integral
        assert abs(occupancy_on_grid(a, d, 100, 0.01) - r["occupancy_integral"]) < 2e-1 * 0.01 * n + 1e-9
        assert close(r["average_backlog"], r["filing_rate"] * r["mean_disposition_time"])  # little
        assert close(r["little_identity_check"], 0) and close(r["required_mean_time"], 2.5 / r["filing_rate"])
        assert close(r["occupancy_integral"], 100 * (r["filing_rate"] * r["mean_disposition_time"]))  # little_backlog
    r = R.court_backlog([0, 1, 2, 4, 5, 7], [3, 2.5, 6, 5, 9, 10], horizon=10)
    assert close(r["average_backlog"], 1.65)  # R
    rc = R.court_backlog([0, 1, 2], [3, None, 12], horizon=10)
    assert (
        rc["n_censored"] == 2
        and close(rc["occupancy_integral"], 3 + 9 + 8)
        and rc["little_identity_check"] != rc["little_identity_check"]
    )
    for args, kw, msg in [
        (([0, 1], [1]), {}, "same positive length"),
        (([-1, 1], [1, 2]), {}, "non-negative"),
        (([0, 1], [1, 2]), {"horizon": 0}, "positive"),
        (([0, 11], [1, 12]), {"horizon": 10}, "inside the horizon"),
        (([2, 1], [1, 2]), {}, "precede"),
        (([0, 1], [1, 2]), {"target_backlog": -1}, "non-negative"),
    ]:
        with pytest.raises(ValueError, match=msg):
            R.court_backlog(*args, **kw)


def test_incapacitation_arithmetic():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(300, seed=18)]
    for k in range(40):
        lam = 0.1 + 20 * u[k]
        q = 0.01 + 0.99 * u[100 + k]
        S = 10 * u[200 + k]
        r = R.incapacitation(lam, q, S)
        rate = float(r["rate"][0])
        ps = float(r["prevented_share"][0])
        mp = float(r["marginal_prevention"][0])
        fbar = 1 / (lam * q)
        assert close(rate, lam * fbar / (fbar + S))  # steady_state_rate
        assert close(ps, 1 - rate / lam) and 0 <= ps < 1 and rate <= lam  # prevented_share_*
        assert float(R.incapacitation(lam, q, S + 1)["rate"][0]) <= rate  # rate_antitone_in_S
        assert float(R.incapacitation(lam, min(1.0, q + 0.1), S)["rate"][0]) <= rate  # rate_antitone_in_q
        assert close(mp, rate - float(R.incapacitation(lam, q, S + 1)["rate"][0])) and mp > 0  # marginal_prevention_*
        assert mp >= float(R.incapacitation(lam, q, S + 1)["marginal_prevention"][0])  # diminishing
    g = R.incapacitation([2, 10], 0.1, 1, shares=[0.8, 0.2])
    assert float(g["prevented_share"][0]) <= float(g["prevented_share"][1])  # high_rate_more_prevented
    assert close(g.attrs["aggregate"]["free_rate"], 3.6) and close(
        g.attrs["aggregate"]["incapacitated_rate"], 0.8 * 2 / 1.2 + 0.2 * 5
    )
    with pytest.raises(ValueError, match="positive"):
        R.incapacitation(0, 0.1, 1)
    with pytest.raises(ValueError, match="summing to one"):
        R.incapacitation([1, 2], 0.1, 1, shares=[0.5, 0.6])


def test_selective_labels_contraction_and_bounds():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(1200, seed=19)]
    n = 300
    risk = u[:n]
    y = [1 if u[300 + i] < risk[i] else 0 for i in range(n)]
    w = [0.5 + 1.5 * u[600 + i] for i in range(n)]
    released = [r < 0.7 for r in risk]
    inner = [r < 0.5 for r in risk]
    r = R.selective_labels(y, released, inner, w)
    assert r["identified"] and r["width"] == 0
    assert close(
        r["rule_rate"], sum(w[i] * y[i] for i in range(n) if inner[i]) / sum(w[i] for i in range(n) if inner[i])
    )  # nested
    assert close(
        r["observed_rate"],
        sum(w[i] * y[i] for i in range(n) if released[i]) / sum(w[i] for i in range(n) if released[i]),
    )
    outer = [r_ < 0.9 for r_ in risk]
    r2 = R.selective_labels(y, released, outer, w)
    assert not r2["identified"]
    assert close(
        r2["width"],
        sum(w[i] for i in range(n) if outer[i] and not released[i]) / sum(w[i] for i in range(n) if outer[i]),
    )
    for fill in ([0] * n, [1] * n, [1 if u[900 + i] < 0.5 else 0 for i in range(n)]):
        yy = [y[i] if released[i] else fill[i] for i in range(n)]
        rate = sum(w[i] * yy[i] for i in range(n) if outer[i]) / sum(w[i] for i in range(n) if outer[i])
        assert r2["bounds"]["lower"] - 1e-12 <= rate <= r2["bounds"]["upper"] + 1e-12  # unobserved_bounds
    y0 = [y[i] if released[i] else 0 for i in range(n)]
    y1 = [y[i] if released[i] else 1 for i in range(n)]
    assert close(
        sum(w[i] * y0[i] for i in range(n) if outer[i]) / sum(w[i] for i in range(n) if outer[i]), r2["bounds"]["lower"]
    )
    assert close(
        sum(w[i] * y1[i] for i in range(n) if outer[i]) / sum(w[i] for i in range(n) if outer[i]), r2["bounds"]["upper"]
    )
    for args, kw, msg in [
        (([0, 1], [True, True], [True]), {}, "equal length"),
        (([0, 2], [True, True], [True, True]), {}, "0/1"),
        (([0, 1], [True, True], [True, True]), {"weights": [-1, 1]}, "non-negative"),
        (([0, 1], [False, False], [True, True]), {}, "positive mass"),
    ]:
        with pytest.raises(ValueError, match=msg):
            R.selective_labels(*args, **kw)
    assert (
        R.selective_labels([0, 1], [True, False], [False, True])["naive_rate"]
        != R.selective_labels([0, 1], [True, False], [False, True])["naive_rate"]
    )


def test_regression_to_the_mean_under_exchangeability():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(1000, seed=20)]
    n = 200
    mu = [0.5 + 6 * u[i] for i in range(n)]
    x1 = [int(mu[i] * (0.5 + u[200 + i])) for i in range(n)]
    x2 = [int(mu[i] * (0.5 + u[400 + i])) for i in range(n)]
    w = [0.5 + 1.5 * u[600 + i] for i in range(n)]
    c = sorted(x1)[int(0.8 * n)]
    r = R.regression_to_mean(x1, x2, c, w)
    assert r["symmetrised_change"] <= 1e-12  # selected_change_nonpos on the symmetrised population
    assert r["low_symmetrised_change"] >= -1e-12  # low_selected_change_nonneg
    assert close(r["excess_over_symmetry"], r["selected_change"] - r["symmetrised_change"])
    X1 = x1 + x2
    X2 = x2 + x1
    W = w + w
    s1 = [v > c for v in X1]
    s2 = [v > c for v in X2]
    assert close(sum(W[i] for i in range(2 * n) if s1[i]), sum(W[i] for i in range(2 * n) if s2[i]))  # exchange_mass
    assert close(
        sum(W[i] * X2[i] for i in range(2 * n) if s1[i]), sum(W[i] * X1[i] for i in range(2 * n) if s2[i])
    )  # exchange_cross
    assert sum(W[i] * (X2[i] - X1[i]) for i in range(2 * n) if s1[i]) <= 1e-12  # selected_change_nonpos
    assert all(
        X1[i] * (int(s1[i]) - int(s2[i])) >= c * (int(s1[i]) - int(s2[i])) - 1e-12 for i in range(2 * n)
    )  # indicator_bound
    rr = R.regression_to_mean(X1, X2, c, W)
    assert close(rr["selected_change"], rr["symmetrised_change"])
    for args, kw, msg in [
        (([1, 2, 3], [1, 2], 1), {}, "equal length"),
        (([1, float("nan")], [1, 2], 1), {}, "NA"),
        (([1, 2, 3], [1, 2, 3], [1, 2]), {}, "single number"),
        (([1, 2, 3], [1, 2, 3], 1), {"weights": [-1, 1, 1]}, "non-negative"),
        (([1, 2, 3], [1, 2, 3], 10), {}, "exceeds the threshold"),
    ]:
        with pytest.raises(ValueError, match=msg):
            R.regression_to_mean(*args, **kw)
    one = R.regression_to_mean([5, 5], [1, 1], 4)
    assert one["mirror_change"] != one["mirror_change"] and one["low_selected_change"] != one["low_selected_change"]


def test_regression_to_the_mean_edge_branches():
    with pytest.raises(ValueError, match="single number"):
        R.regression_to_mean([1, 2, 3], [1, 2, 3], float("nan"))
    r = R.regression_to_mean([5, 6], [7, 8], 4)
    assert r["low_symmetrised_change"] != r["low_symmetrised_change"]


# ----------------------------------------------------------------- P13 HKSJ (P13HKSJ.lean)


def test_meta_hksj_theorems_and_parity():
    est = [-0.25, -0.10, -0.40, 0.05, -0.30]
    v = [0.010, 0.020, 0.015, 0.030, 0.012]
    h = R.meta_hksj(est, v)
    # R: tau2 0.00696774193548387, q 1.08617755964300433, hksj se 0.07005678381727683, wald se 0.06722019728093469
    assert close(h["tau2"], 0.00696774193548387, 1e-10)
    assert close(h["q"], 1.08617755964300433, 1e-10)
    assert close(h["hksj"]["se"], 0.07005678381727683, 1e-10)
    assert close(h["wald"]["se"], 0.06722019728093469, 1e-10)
    w = [1 / (vi + h["tau2"]) for vi in v]
    mu = sum(wi * yi for wi, yi in zip(w, est)) / sum(w)
    Q = sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, est))
    assert close(h["Q"], Q, 1e-10) and close(h["q"], Q / 4, 1e-10)
    assert close(h["hksj"]["variance"], h["q"] / sum(w), 1e-10)
    assert h["wider_than_wald"] == (h["q"] >= 1) == (h["hksj"]["variance"] >= h["wald"]["variance"])  # hksj_wider_iff
    assert not h["degenerate"] and not h["equal_weights"] and h["hksj"]["df"] == 4
    r = R.meta_hksj(est, v, tau2="REML")
    # R: tau2 0.00290803591431826, q 1.27119896482387307, hksj se 0.06805979694728266
    assert close(r["tau2"], 0.00290803591431826, 1e-9)
    assert close(r["q"], 1.27119896482387307, 1e-9)
    assert close(r["hksj"]["se"], 0.06805979694728266, 1e-9)
    assert r["tau2_method"] == "REML"

    # REML maximises the restricted log-likelihood on a grid
    def ll(t):
        ww = [1 / (vi + t) for vi in v]
        m = sum(a * b for a, b in zip(ww, est)) / sum(ww)
        return (
            -0.5 * sum(math.log(vi + t) for vi in v)
            - 0.5 * math.log(sum(ww))
            - 0.5 * sum(a * (b - m) ** 2 for a, b in zip(ww, est))
        )

    assert ll(r["tau2"]) >= max(ll(i * 1e-4) for i in range(2001)) - 1e-9
    assert R.meta_hksj([0.10, 0.11, 0.09, 0.10], [0.05] * 4, tau2="REML")["tau2"] == 0
    # Q_eq_zero_iff
    d = R.meta_hksj([0.2, 0.2, 0.2, 0.2], [0.01, 0.02, 0.03, 0.04])
    assert d["degenerate"] and d["hksj"]["se"] == 0 and d["hksj"]["ci"] == {"lower": 0.2, "upper": 0.2}
    # hksj_equal_weights: equal variances give s^2/k for DL and REML
    y = [0.3, -0.1, 0.7, 0.2, -0.4, 0.5]
    for method in ("DL", "REML"):
        e = R.meta_hksj(y, [0.04] * 6, tau2=method)
        ybar = sum(y) / 6
        s2 = sum((yi - ybar) ** 2 for yi in y) / 5
        assert (
            e["equal_weights"] and close(e["hksj"]["variance"], s2 / 6, 1e-10) and close(e["t_variance"], s2 / 6, 1e-10)
        )
        assert close(e["estimate"], ybar, 1e-12)
    # hksj_wider_iff on both sides: the example has q > 1; near-identical sites with large variances have q < 1
    narrow = R.meta_hksj([0.10, 0.11, 0.09, 0.10], [0.05, 0.06, 0.04, 0.05])
    assert narrow["q"] < 1 and not narrow["wider_than_wald"] and narrow["hksj"]["variance"] < narrow["wald"]["variance"]
    assert h["q"] > 1 and h["wider_than_wald"] and h["hksj"]["variance"] > h["wald"]["variance"]


# ----------------------------------------------------------------- P16 censoring (P16Censoring.lean)


def test_backlog_censoring_theorems_and_parity():
    b = R.backlog_censoring([30, 45, 60, 90, 120], [100, 150, 200])
    assert (b["disposed_mean"], b["lower_bound"], b["bias_lower"], b["understates"]) == (69.0, 99.375, 30.375, True)
    assert b["upper_bound"] == math.inf and b["n"] == 5 and b["m"] == 3 and b["pending_age"] == 150.0
    t, a = [30.0, 45.0, 60.0, 90.0, 120.0], [100.0, 150.0, 200.0]
    assert close(b["lower_bound"] - b["disposed_mean"], (3 / 8) * (150 - 69))  # lower_bound_sub
    for extra in (0.0, 10.0, 500.0):
        u = [x + extra for x in a]
        truth = (sum(t) + sum(u)) / 8
        assert truth >= b["lower_bound"] - 1e-12  # true_mean_ge
        assert truth - b["disposed_mean"] >= b["bias_lower"] - 1e-12  # bias_lower
        assert truth >= b["disposed_mean"]  # disposed_understates
    # no_upper_bound
    for level in (100.0, 1e3, 1e6):
        K = max((level * 8 - sum(t) - sum(a)) / 3 + 1, 0)
        assert (sum(t) + sum(x + K for x in a)) / 8 > level
    c = R.backlog_censoring([100, 200], [10])
    assert not c["understates"] and c["bias_lower"] < 0


def test_judge_monotonicity_nested_and_crossing():
    import random

    rng = random.Random(7)
    thr = {"J1": 0.2, "J2": 0.3, "J3": 0.4, "J4": 0.5, "J5": 0.6, "J6": 0.7}
    judge = [f"J{1 + i % 6}" for i in range(6000)]
    group = ["abc"[i // 2000] for i in range(6000)]
    u = [rng.random() for _ in range(6000)]
    d = [int(ui < thr[j]) for ui, j in zip(u, judge)]
    r = R.judge_monotonicity(judge, d, group)
    assert r["summary"]["significant"] == 0 and r["summary"]["comparisons"] == 45
    assert all(g["slope"] > 0 for g in r["by_group"]) and r["judges"][0]["judge"] == "J6"
    c = r["comparisons"][0]
    pp = (c["group_rate_stricter"] * c["n_stricter"] + c["group_rate_lenient"] * c["n_lenient"]) / (
        c["n_stricter"] + c["n_lenient"]
    )
    z = (c["group_rate_lenient"] - c["group_rate_stricter"]) / math.sqrt(
        pp * (1 - pp) * (1 / c["n_stricter"] + 1 / c["n_lenient"])
    )
    assert close(c["z"], z)
    d2 = [int(ui < 0.05) if (j == "J6" and g == "c") else di for ui, j, g, di in zip(u, judge, group, d)]
    r2 = R.judge_monotonicity(judge, d2, group)
    sig = [x for x in r2["comparisons"] if x["significant"]]
    assert sig and all("J6" in (x["stricter"], x["lenient"]) for x in sig)
    sl = {g["group"]: g["slope"] for g in r2["by_group"]}
    assert sl["c"] < min(sl["a"], sl["b"])
    for x in r2["comparisons"]:  # Holm: adjusted at least raw, at most one
        assert x["p_value"] - 1e-15 <= x["p_adjusted"] <= 1.0


def test_judge_monotonicity_errors():
    for args, kw, msg in [
        (([1, 2, 3], [0, 1], [1, 2, 3]), {}, "equal length"),
        ((["a", None], [0, 1], [1, 2]), {}, "missing"),
        ((["a", "b"], [0, 2], [1, 2]), {}, "0/1"),
        ((["a", "b"], [0, 1], [1, 2]), {"min_n": 0}, "min_n"),
        ((["a", "b"], [0, 1], [1, 2]), {"alpha": 1}, "alpha"),
        ((["a", "b"], [0, 1], [1, 2]), {"adjust": "bh"}, "holm"),
    ]:
        with pytest.raises(ValueError, match=msg):
            R.judge_monotonicity(*args, **kw)


def test_backlog_published_little_and_bands_parity():
    r = R.backlog_published(
        21265, 30, 148904, 175, band_shares=[[0.808, 0.105, 0.034, 0.053]], band_floors=[0, 304, 456, 548]
    )[0]
    lam = 21265 / 30
    assert close(r["filing_rate"], lam) and close(r["implied_mean_days"], 148904 / lam)
    assert close(r["ratio"], (148904 / lam) / 175) and close(r["little_backlog"], lam * 175)
    assert close(r["band_floor_bound"], 0.105 * 304 + 0.034 * 456 + 0.053 * 548)  # band_floor_shares
    # band_floor_le_mean on a docket whose cases sit at or above their floors
    floors = [0, 304, 456, 548]
    band = [0, 0, 1, 2, 3, 3, 1, 0]
    dur = [floors[b] + 7.0 * (i + 1) for i, b in enumerate(band)]
    sh = [band.count(k) / len(band) for k in range(4)]
    bb = R.backlog_published(1, 1, 0, sum(dur) / len(dur), band_shares=[sh], band_floors=floors)[0]
    assert bb["band_floor_bound"] <= sum(dur) / len(dur)
    assert close(bb["band_floor_bound"], sum(floors[b] for b in band) / len(band))
    w = R.backlog_published([300, 300, 300], 30, 450, 45)
    assert [round(x["ratio"], 12) for x in w] == [1.0, 1.0, 1.0] and "band_floor_bound" not in w[0]


def test_backlog_published_errors():
    for args, msg in [
        ((float("nan"), 30, 1, 1), "finite"),
        ((0, 30, 1, 1), "positive"),
        ((1, 30, -1, 1), "non-negative"),
        (([1, 2, 3], [1, 2], 1, 1), "length 1 or 3"),
    ]:
        with pytest.raises(ValueError, match=msg):
            R.backlog_published(*args)
    for kw, msg in [
        (dict(band_shares=[[0.5, 0.5], [0.5, 0.5]]), "one row per window"),
        (dict(band_shares=[[0.5, 0.5]]), "one lower edge"),
        (dict(band_shares=[[-0.5, 1.5]], band_floors=[0, 1]), "non-negative"),
        (dict(band_shares=[[0.2, 0.2]], band_floors=[0, 1]), "sum to 1"),
        (dict(band_shares=[[0.5, 0.5]], band_floors=[5, 1]), "increasing"),
    ]:
        with pytest.raises(ValueError, match=msg):
            R.backlog_published(1, 1, 1, 1, **kw)
    with pytest.raises(ValueError, match="finite"):
        R.backlog_published("a", 1, 1, 1)


# ----------------------------------------------------------------- P17 replacement and desistance (P17Replacement.lean)


def test_incapacitation_career_theorems_and_parity():
    lam = [12, 10, 8, 6, 5, 4, 3, 2, 2, 1]
    r = R.incapacitation_career(lam, t0=2, S=3, replacement=0.25)
    assert (r["prevented"], r["prevented_net"], r["upper"], r["lower"], r["later"]) == (19.0, 14.25, 24.0, 12.0, 15.0)
    assert r["factor"] == 0.75 and r["replacement"] == 0.25
    for t0 in range(0, 6):
        for S in range(1, 4):
            if len(lam) < t0 + S + 1:
                continue
            out = R.incapacitation_career(lam, t0, S)
            assert close(out["prevented"], sum(lam[t0 : t0 + S]))
            assert out["prevented"] <= out["upper"] + 1e-12  # prevented_le_const
            assert out["prevented"] >= out["lower"] - 1e-12  # prevented_ge_const
            if not math.isnan(out["later"]):
                assert out["later"] <= out["prevented"] + 1e-12  # later_sentence_prevents_less
            net = R.incapacitation_career(lam, t0, S, replacement=0.4)
            assert net["prevented_net"] <= out["prevented_net"] + 1e-12  # replaced_antitone
            assert net["prevented_net"] <= out["upper"] + 1e-12  # prevented_net_le
    c = R.incapacitation_career([3] * 6, t0=1, S=4)
    assert (
        c["prevented"] == 12.0 and c["upper"] == 12.0 and c["lower"] == 12.0 and math.isnan(c["later"])
    )  # constant_rate
    assert R.incapacitation_career([3] * 6, t0=1, S=4, replacement=1)["prevented_net"] == 0.0  # replaced_full


# ----------------------------------------------------------------- P19 shrinkage (P19Shrinkage.lean)


def _noise_law(theta, e, w):
    """Project e onto the orthogonal complement of {1, theta} under w (so sum w e = 0 and sum w theta e = 0)."""
    s_w = sum(w)
    s_wt = sum(a * b for a, b in zip(w, theta))
    s_wtt = sum(a * b * b for a, b in zip(w, theta))
    s_we = sum(a * b for a, b in zip(w, e))
    s_wte = sum(a * b * c for a, b, c in zip(w, theta, e))
    det = s_w * s_wtt - s_wt**2
    alpha = (s_wtt * s_we - s_wt * s_wte) / det
    beta = (s_w * s_wte - s_wt * s_we) / det
    return [ei - alpha - beta * ti for ei, ti in zip(e, theta)]


def test_shrinkage_theorems_and_parity():
    sl = R.shrinkage_loss([10, 20, 30, 40], [3, -3, -3, 3], B=0.5)
    assert (sl["loss"], sl["closed_form"], sl["loss_raw"], sl["noise_law"]) == (134.0, 134.0, 36.0, True)
    assert close(sl["B_star"], 0.0671641791044776, 1e-12) and close(sl["loss_star"], 33.5820895522388057, 1e-12)
    assert close(sl["Se"], 36.0) and close(sl["Stheta"], 500.0)
    theta = [3.0, 11.0, 7.0, 25.0, 14.0, 9.0, 31.0]
    w = [1.0, 0.5, 2.0, 1.5, 1.0, 0.8, 1.2]
    e = _noise_law(theta, [2.0, -1.0, 4.0, -3.0, 0.5, 1.5, -2.5], w)
    assert abs(sum(a * b for a, b in zip(w, e))) < 1e-9 and abs(sum(a * b * c for a, b, c in zip(w, theta, e))) < 1e-9
    base = R.shrinkage_loss(theta, e, w, B=0.3)
    assert base["noise_law"]
    Se, St = base["Se"], base["Stheta"]
    Bstar = Se / (Se + St)
    star = R.shrinkage_loss(theta, e, w, B=Bstar)
    assert close(star["loss"], star["closed_form"], 1e-9)  # loss_eq
    assert close(star["loss"], Se * St / (Se + St), 1e-9) and close(
        star["loss_star"], Se * St / (Se + St), 1e-12
    )  # loss_bstar_eq
    assert star["loss_star"] <= star["loss_raw"]  # loss_bstar_le_raw
    assert 0 <= Bstar <= 1  # bstar_mem
    for B in (0.0, 0.1, 0.5, 0.9, 1.0, Bstar + 0.05):
        lb = R.shrinkage_loss(theta, e, w, B=B)
        assert close(lb["loss"], (1 - B) ** 2 * Se + B**2 * St, 1e-9)  # loss_eq
        assert lb["loss"] >= star["loss"] - 1e-9  # loss_min
    assert close(R.shrinkage_loss(theta, e, w, B=0)["loss"], Se, 1e-9)  # loss_raw
    off = R.shrinkage_loss([1, 2, 3], [1, 1, 1], B=0.5)
    assert not off["noise_law"] and not close(off["loss"], off["closed_form"])
    z = R.shrinkage_loss([1, 1], [0, 0], B=0.3)
    assert z["B_star"] == 0 and z["loss_star"] == 0
    # hotspot_shrinkage parity: R B 0.132686449283538, fall_top 2.869344465756516, shrunk_top 37.130655534243481
    y = [40, 12, 9, 25, 7, 31, 5, 18]
    s = R.hotspot_shrinkage(y, noise_variance=sum(y) / 8)
    assert close(s["B"], 0.132686449283538, 1e-12) and close(s["predicted_fall"][0], 2.869344465756516, 1e-12)
    assert close(s["shrunk"][0], 37.130655534243481, 1e-12) and s["total_variance"] == 138.484375
    ybar = sum(y) / 8
    assert all(close(yi - si, s["B"] * (yi - ybar), 1e-10) for yi, si in zip(y, s["shrunk"]))  # predicted_fall
    assert s["B"] == s["B_star"] and 0 <= s["B"] <= 1
    Se2 = s["noise_variance"] * 8
    St2 = s["signal_variance"] * 8
    assert close(s["B_star"], Se2 / (Se2 + St2), 1e-12)
    fixed = R.hotspot_shrinkage(y, noise_variance=sum(y) / 8, weights=[1, 2, 1, 2, 1, 2, 1, 2], B=0.25)
    wm = sum(a * b for a, b in zip([1, 2, 1, 2, 1, 2, 1, 2], y)) / 12
    assert fixed["B"] == 0.25 and all(close(yi - si, 0.25 * (yi - wm), 1e-10) for yi, si in zip(y, fixed["shrunk"]))
    assert R.hotspot_shrinkage([0, 0, 0], noise_variance=0)["B"] == 0


# ----------------------------------------------------------------- P14 slope test (P14Slope.lean)


def test_judge_slope_theorems_and_parity():
    judge = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    d = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
    y = [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]
    s = R.judge_slope_test(judge, d, y)
    assert s["judges"]["judge"] == ["A", "B", "C"] and s["judges"]["n"] == [6, 6, 6]
    assert all(close(a, b) for a, b in zip(s["judges"]["propensity"], [2 / 6, 4 / 6, 5 / 6]))
    assert all(close(a, b) for a, b in zip(s["judges"]["outcome"], [2 / 6, 3 / 6, 5 / 6]))
    assert s["pairs"]["violation"] == [False, False, True] and s["violations"] == 1 and not s["monotone_consistent"]
    assert all(close(a, b) for a, b in zip(s["pairs"]["late"], [0.5, 1.0, 2.0]))
    assert (s["lo"], s["hi"]) == (0.0, 1.0)
    # nested judges on one population: propensity_mono, outcome_diff, slope_bound, no violation
    n, J = 25, 4
    w = [1 + (i % 3) * 0.5 for i in range(n)]
    y0 = [((i * 7) % 11) / 10 for i in range(n)]
    y1 = [((i * 5 + 3) % 11) / 10 for i in range(n)]
    sev = [((i * 13) % 17) / 17 for i in range(n)]
    cuts = [0.8, 0.6, 0.4, 0.2]
    dmat = [[1.0 if sev[i] > c else 0.0 for c in cuts] for i in range(n)]
    jj = [str(j) for j in range(J) for _ in range(n)]
    dd = [dmat[i][j] for j in range(J) for i in range(n)]
    yy = [y1[i] if dmat[i][j] else y0[i] for j in range(J) for i in range(n)]
    ww = [w[i] for _ in range(J) for i in range(n)]
    t = R.judge_slope_test(jj, dd, yy, weights=ww, lo=0, hi=1)
    assert t["violations"] == 0 and t["monotone_consistent"]
    P = t["judges"]["propensity"]
    assert all(P[i] <= P[i + 1] + 1e-15 for i in range(J - 1))  # propensity_mono
    W = sum(w)
    for r in range(len(t["pairs"]["j"])):
        j, k = int(t["pairs"]["j"][r]), int(t["pairs"]["k"][r])
        marginal = sum(w[i] * (dmat[i][k] - dmat[i][j]) * (y1[i] - y0[i]) for i in range(n)) / W
        assert close(t["pairs"]["dY"][r], marginal, 1e-10)  # outcome_diff
        assert abs(t["pairs"]["dY"][r]) <= t["pairs"]["dP"][r] + 1e-12  # slope_bound
        if t["pairs"]["dP"][r] > 0:
            assert close(t["pairs"]["late"][r], t["pairs"]["dY"][r] / t["pairs"]["dP"][r])
    one = R.judge_slope_test(["A"] * 3, [0, 1, 1], [1, 0, 1])
    assert one["pairs"] is None and one["violations"] == 0


# ----------------------------------------------------------------- P15 DFL reweighting (P15Reweight.lean)


def test_dfl_reweight_theorems_and_parity():
    g = [True] * 6 + [False] * 6
    x = ["a", "a", "a", "a", "b", "b", "a", "a", "b", "b", "b", "b"]
    y = [10, 12, 11, 13, 20, 22, 8, 9, 15, 16, 14, 17]
    r = R.dfl_reweight(g, x, y)
    # R: mean_1 14.66666666666667, mean_0 13.16666666666667, cf 10.83333333333333, structure 3.83333333333333, composition -2.33333333333333
    assert close(r["mean_1"], 14.66666666666667, 1e-12) and close(r["mean_0"], 13.16666666666667, 1e-12)
    assert close(r["counterfactual"], 10.83333333333333, 1e-12)
    assert close(r["structure"], 3.83333333333333, 1e-12) and close(r["composition"], -2.33333333333333, 1e-12)
    assert r["psi"] == {"x": ["a", "b"], "mass_1": [4.0, 2.0], "mass_0": [2.0, 4.0], "psi": [2.0, 0.5]}
    assert r["max_composition_gap"] == 0.0 and r["reweighted_mass"] == r["mass_1"] == 6.0  # reweighted_mass
    assert close(r["mean_1"] - r["mean_0"], r["structure"] + r["composition"])  # decomposition
    # counterfactual_outcome: group-0 outcomes a function of x
    mu0 = {"a": 8.0, "b": 15.0}
    y2 = y[:6] + [mu0[v] for v in x[6:]]
    c = R.dfl_reweight(g, x, y2)
    assert close(c["counterfactual"], (4 / 6) * 8 + (2 / 6) * 15)
    # reweighting_matches with weights and three values, any h
    g3 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    x3 = ["a", "b", "c", "a", "b", "a", "a", "b", "c", "c", "b", "a"]
    w3 = [1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 1.0, 0.5, 1.0, 1.5, 2.0, 1.0]
    y3 = [float(i) for i in range(12)]
    r3 = R.dfl_reweight(g3, x3, y3, weights=w3)
    psi = dict(zip(r3["psi"]["x"], r3["psi"]["psi"]))
    for h in ({"a": 1.0, "b": -2.0, "c": 0.5}, {"a": 3.0, "b": 3.0, "c": -1.0}):
        lhs = sum(psi[x3[i]] * w3[i] * h[x3[i]] for i in range(12) if not g3[i])
        rhs = sum(w3[i] * h[x3[i]] for i in range(12) if g3[i])
        assert close(lhs, rhs, 1e-10)
    assert r3["max_composition_gap"] < 1e-12
    with pytest.raises(ValueError, match="common support fails at x = b"):
        R.dfl_reweight([True, True, False], ["a", "b", "a"], [1, 2, 3])


# ----------------------------------------------------------------- P1 Le Cam (P1LeCam.lean)


def test_two_point_bound_theorems_and_parity():
    p = [0.0625, 0.25, 0.375, 0.25, 0.0625]
    q = [0.0256, 0.1536, 0.3456, 0.3456, 0.1296]
    b = R.two_point_bound(p, q, theta_p=2, theta_q=3, estimator=[0, 1.25, 2.5, 3.75, 5])
    # R: tv 0.1627, bound 0.41865, minimax 1.125, risk_q 1.0368, sum_min 0.8373
    assert close(b["tv"], 0.1627, 1e-12) and close(b["bound"], 0.41865, 1e-12)
    assert (
        close(b["minimax_risk"], 1.125, 1e-12)
        and close(b["risk_q"], 1.0368, 1e-12)
        and close(b["sum_min"], 0.8373, 1e-12)
    )
    assert b["satisfied"] and b["delta"] == 1
    assert close(b["sum_min"], 1 - b["tv"])  # sum_min
    assert 0 <= b["tv"] <= 1  # tv_nonneg, tv_le_one
    assert b["risk_p"] + b["risk_q"] >= b["delta"] * (1 - b["tv"]) - 1e-12  # two_point
    assert b["minimax_risk"] >= b["bound"] - 1e-12  # minimax
    for T in ([2.5] * 5, [0, 0, 0, 0, 0], [3, 2, 3, 2, 3], [-10, 10, -10, 10, 0]):
        e = R.two_point_bound(p, q, 2, 3, estimator=T)
        assert e["risk_p"] + e["risk_q"] >= 1 - e["tv"] - 1e-12 and e["satisfied"]
    no = R.two_point_bound(p, q, 2, 3)
    assert "risk_p" not in no and close(no["bound"], 0.41865, 1e-12)
    same = R.two_point_bound([0.5, 0.5], [0.5, 0.5], 0, 1)
    assert same["tv"] == 0 and same["bound"] == 0.5
    far = R.two_point_bound([1, 0], [0, 1], 0, 1)
    assert far["tv"] == 1 and far["bound"] == 0


# ----------------------------------------------------------------- P9 regimes, P11 cutoff, P18 chain


def test_recording_regimes_and_second_source():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(60, seed=9)]
    for k in range(10):
        m = 2 + k % 4
        r = [500 * u[k * 5 + i] for i in range(m)]
        eps = 0.05 + 0.85 * u[50 + k]
        two = R.recording_regimes(r, cuffed_share=eps)
        rec_a = [sum(two["reclassification"]["M"][i][j] * r[j] for j in range(m)) for i in range(m)]
        cc = list(two["cuffing"]["counts"].values())
        rec_b = [sum(two["cuffing"]["M"][i][j] * cc[j] for j in range(m)) for i in range(m)]
        assert all(close(a, b) for a, b in zip(rec_a, r)) and all(close(a, b) for a, b in zip(rec_b, r))
        assert close(two["reclassification"]["true_total"], sum(r))
        assert close(two["cuffing"]["true_total"], sum(r) / (1 - eps))  # regimes_not_identified
        assert two["cuffing"]["true_total"] > two["reclassification"]["true_total"]
        assert not two["identified_from_series"]
        assert R.recording_regimes(r, eps, true_total=sum(r))["second_source"]["regime"] == "reclassification"
        sec = R.recording_regimes(r, eps, true_total=two["cuffing"]["true_total"])["second_source"]
        assert sec["regime"] == "cuffing" and close(sec["dropped_mass"], two["cuffing"]["true_total"] - sum(r))
        assert R.recording_regimes(r, eps, true_total=sum(r) / 2)["second_source"]["regime"] == "inconsistent"
    with pytest.raises(ValueError, match="non-negative"):
        R.recording_regimes([1, -1])
    with pytest.raises(ValueError, match="cuffed_share"):
        R.recording_regimes([1, 2], cuffed_share=1)


def test_imbens_manski_cutoff_exists_and_is_unique():
    from morie.fn._rng import random_uniform
    from morie.fn._stats_core import norm

    u = [float(v) for v in random_uniform(40, seed=11)]
    for k in range(20):
        delta, alpha = 6 * u[k], 0.01 + 0.29 * u[20 + k]

        def f(c):
            return float(norm.cdf(c + delta)) - float(norm.cdf(-c)) - (1 - alpha)

        z1, z2 = float(norm.ppf(1 - alpha)), float(norm.ppf(1 - alpha / 2))
        assert f(z1 - 1e-9) <= 1e-12 and f(z2 + 1e-9) >= -1e-12  # im_cutoff_exists (a sign change)
        r = R.bounds_confidence(0, delta * 0.1, 0.1, 0.1, 1 - alpha)
        assert close(f(r["cutoff"]), 0, 1e-9)
        assert all(abs(f(r["cutoff"] + d)) > 1e-6 for d in (-1, -0.1, 0.1, 1))  # im_cutoff_unique
        assert "Research.P11.im_cutoff_exists" in r["theorems"]


def test_selective_labels_judges_chain():
    from morie.fn._rng import random_uniform

    u = [float(v) for v in random_uniform(1500, seed=23)]
    n = 250
    risk = u[:n]
    y = [1 if u[n + i] < risk[i] else 0 for i in range(n)]
    w = [0.5 + 1.5 * u[2 * n + i] for i in range(n)]
    for k in range(10):
        cuts = sorted(0.3 + 0.65 * u[750 + 5 * k + j] for j in range(4))
        judges = [[r < ct for r in risk] for ct in cuts]
        rule = [r < 0.2 + 0.8 * u[800 + k] for r in risk]
        res = R.selective_labels_judges(y, judges, rule, w)
        assert res["nested"]
        assert all(b <= a + 1e-12 for a, b in zip(res["widths"], res["widths"][1:]))  # chain_width_antitone
        for j in range(4):
            width_j = sum(w[i] for i in range(n) if rule[i] and not judges[j][i]) / sum(
                w[i] for i in range(n) if rule[i]
            )
            assert close(res["judges"][j]["width"], width_j)
            assert (res["judges"][j]["width"] == 0) == (not any(rule[i] and not judges[j][i] for i in range(n)))
        inside_union = not any(rule[i] and not any(judges[j][i] for j in range(4)) for i in range(n))
        assert res["judges"][3]["identified"] == inside_union  # chain_contraction
        assert res["best"] == res["judges"][min(range(4), key=lambda j: res["widths"][j])]["judge"]
    nonnested = R.selective_labels_judges(
        y, [[r < 0.6 for r in risk], [r > 0.3 for r in risk]], [r < 0.5 for r in risk]
    )
    assert not nonnested["nested"] and close(nonnested["width"], min(nonnested["widths"]))
    with pytest.raises(ValueError, match="one entry per defendant"):
        R.selective_labels_judges(y, [[True] * 10], [r < 0.5 for r in risk])


# ----------------------------------------------------------------- 2026-10-12 closures (R/ledger_1_4_3.R)


def test_veil_of_darkness_parity():
    # R: morie_veil_of_darkness(c(white = 400, black = 200), c(white = 300, black = 100))$bias_ratio -> 1, 1.5
    r = R.veil_of_darkness({"white": 400, "black": 200}, {"white": 300, "black": 100})
    assert close(r["bias_ratio"]["white"], 1) and close(r["bias_ratio"]["black"], 1.5)
    assert r["sign"] == {"white": "lower", "black": "higher"}
    assert close(r["share_day"]["black"], 200 / 600) and close(r["share_night"]["black"], 100 / 400)
    # odds_ratio_identifies_bias on a synthetic road: exposures and dark propensities cancel
    E, rr, d = {"w": 1000, "b": 400}, {"w": 0.03, "b": 0.05}, {"w": 0.1, "b": 0.5}
    night = {g: E[g] * rr[g] for g in E}
    day = {g: night[g] * (1 + d[g]) for g in E}
    v = R.veil_of_darkness(day, night)
    assert close(v["bias_ratio"]["b"], 1.5 / 1.1)
    k = {"w": 1.2, "b": 0.8}
    v2 = R.veil_of_darkness({g: day[g] * k[g] for g in E}, night, exposure_shift=k)
    assert close(v2["bias_ratio_adjusted"]["b"], 1.5 / 1.1)  # exposure_shift_confounds, divided out
    v3 = R.veil_of_darkness({g: day[g] * 2 for g in E}, night)
    assert close(v3["bias_ratio"]["b"], v["bias_ratio"]["b"])  # bias_not_absolute
    assert R.veil_of_darkness(night, night)["sign"]["b"] == "equal"
    with pytest.raises(ValueError):
        R.veil_of_darkness(day, {"w": 1})


def test_dark_figure_no_interaction_parity():
    # R: morie_dark_figure_no_interaction(...)$missing 400, $total 738
    cells = {"n100": 120, "n010": 80, "n001": 60, "n110": 30, "n101": 24, "n011": 16, "n111": 8}
    r = R.dark_figure_no_interaction(cells, interaction=[-0.5, 0.0, 0.5])
    assert close(r["missing"], 400) and close(r["total"], 738)
    assert close(r["sensitivity"][0]["missing"], 400 * math.exp(0.5))  # missing_cell_with_interaction
    assert close(r["sensitivity"][1]["total"], 738)
    with pytest.raises(ValueError):
        R.dark_figure_no_interaction({"n100": 1})


def test_shrinkage_factor_error_parity():
    # R: morie_shrinkage_factor_error(c(3, 5, 9, 2, 7, 4), 5, total_variance = 6):
    #    B_hat 0.882352941176471, S_y_hat 5.66666666666667, S_y_se 2.91753320460968, error 0.0490196078431372,
    #    B_interval lower 0.434717095850777
    r = R.shrinkage_factor_error([3, 5, 9, 2, 7, 4], noise_variance=5, total_variance=6)
    assert close(r["B_hat"], 0.882352941176471, 1e-12) and close(r["S_y_hat"], 5.66666666666667, 1e-12)
    assert close(r["S_y_se"], 2.91753320460968, 1e-12) and close(r["error"], 0.0490196078431372, 1e-12)
    assert close(r["B_interval"][0], 0.434717095850777, 1e-12)
    assert r["bound_holds"]
    # bhat_error: the algebraic form
    assert close(r["error"], 5 * (6 - r["S_y_hat"]) / (6 * r["S_y_hat"]))
    with pytest.raises(ValueError):
        R.shrinkage_factor_error([2, 2, 2], 1)


def test_concentration_null_bound_parity():
    # R: gini 0.666666666666667, cv_bound 0.988826464946089, null_bound 0.483045891539648
    r = R.concentration_null_bound([0, 0, 1, 2, 9, 0, 3], n_sim=300, seed=3)
    assert close(r["gini"], 0.666666666666667, 1e-12)
    assert close(r["cv_bound"], 0.988826464946089, 1e-12)
    assert close(r["null_bound"], 0.483045891539648, 1e-12)
    assert r["gini"] <= r["cv_bound"] and 0 <= r["p_value"] <= 1
    # gini_le_cv on random counts
    import random as _r

    g = _r.Random(4)
    for _ in range(30):
        x = [sum(1 for _ in range(int(g.random() * 12))) for _ in range(g.randint(4, 20))]
        if sum(x) == 0:
            continue
        m = sum(x) / len(x)
        v = sum((xi - m) ** 2 for xi in x) / len(x)
        assert R.concentration_gini(x) <= math.sqrt(v / 2) / m + 1e-12


def test_deterrence_logcost_and_wald_parity():
    p, s, c = [0.2, 0.5, 0.8, 0.4], [2, 5, 3, 8], [1, 4, 2, 6]
    y = [2 - 0.8 * (math.log(pi) + math.log(si) - 0.1 * ci) for pi, si, ci in zip(p, s, c)]
    r = R.deterrence_logcost(p, s, c, y, rho=0.1)
    # R: coefficients 2, -0.8, -0.8, 0.08; implied_discount 0.1; expected_cost[1] 0.3619349672143838
    for k, v in (("(Intercept)", 2), ("log_p", -0.8), ("log_s", -0.8), ("c", 0.08)):
        assert close(r["coefficients"][k], v, 1e-9)
    assert close(r["implied_discount"], 0.1, 1e-9)
    assert close(r["expected_cost"][0], 0.3619349672143838, 1e-12)
    assert close(r["log_expected_cost"][0], math.log(r["expected_cost"][0]))
    assert not R.deterrence_logcost(p, [5, 5, 5, 5], c)["identified"]
    with pytest.raises(ValueError):
        R.deterrence_logcost(p, [5, 5, 5, 5], c, y)
    # R: morie_wald_iv(c(1,3,2,4), c(0,1,0,1), c(0,1,0,1)) -> beta 2, cov_zy 0.5, cov_zp 0.25, alpha 1.5
    w = R.wald_iv([1, 3, 2, 4], [0, 1, 0, 1], [0, 1, 0, 1])
    assert close(w["beta"], 2) and close(w["cov_zy"], 0.5) and close(w["cov_zp"], 0.25) and close(w["alpha"], 1.5)
    dead = R.wald_iv([1, 3, 2, 4], [0, 1, 0, 1], [1, 1, 1, 1])
    assert not dead["relevant"] and math.isnan(dead["beta"])


def test_rif_quantile_parity():
    # R: morie_rif_quantile(1:8, 0.5, q = 4, f = 0.125): mean_rif 4, recentering 0, F_q 0.5
    r = R.rif_quantile([1, 2, 3, 4, 5, 6, 7, 8], 0.5, q=4, f=0.125)
    assert close(r["mean_rif"], 4) and close(r["recentering"], 0) and close(r["F_q"], 0.5)
    assert close(r["rif"][0], 4 + (0.5 - 1) / 0.125)
    # off the quantile: rif_quantile_avg
    r2 = R.rif_quantile([1, 2, 3, 4, 5, 6, 7, 8], 0.5, q=2, f=0.3)
    assert close(r2["mean_rif"], 2 + (0.5 - 2 / 8) / 0.3)
    # the default evaluation point is the weighted sample quantile (R: q = 4 for 0.4 on 1..8, 11); the kernel
    # density value is R's binned density() there (0.108405051452871) against the exact Gaussian sum here
    r3 = R.rif_quantile([1, 2, 3, 4, 5, 6, 7, 8, 11], 0.4)
    assert close(r3["q"], 4) and close(r3["f"], 0.108405051452871, 2e-3)
    g = R.rif_gap([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8, 10, 12, 14, 16], 0.5)
    assert close(g["gap"], g["quantile_gap"] + g["recentering_gap"])  # rif_gap
    with pytest.raises(ValueError):
        R.rif_quantile([1, 2, 3], 1.2)


def test_contagion_moment_witness_and_panel_cov_parity():
    w = R.contagion_moment_witness(lambda m, v: v / m)
    assert w["mean"] == (1.0, 1.0) and w["variance"] == (1.0, 1.0) and w["statistic"] == (1.0, 1.0)
    assert close(w["zero_probability"][0], 0.5) and close(w["zero_probability"][1], 1 / 3)
    # R: morie_age_crime_type_cov(c(0.2, 0.3, 0.5), c(1, 2, 4), c(3, 1, 2))$cross_cov -0.12
    t = R.age_crime_type_cov([0.2, 0.3, 0.5], [1, 2, 4], [3, 1, 2])
    assert close(t["cross_cov"], -0.12, 1e-12) and not t["one_type"]
    assert close(R.age_crime_type_cov([0.5, 0.5], [1, 3], [3, 1])["cross_cov"], -1)  # two_type_cov
    assert R.age_crime_type_cov([0.5, 0.5], [2, 2], [2, 2])["one_type"]
    # R: morie_age_crime_panel_cov(c(1,3,0,2,5,1), c(2,0,4,1,1,3))$cross_cov -1.66666666666667
    pc = R.age_crime_panel_cov([1, 3, 0, 2, 5, 1], [2, 0, 4, 1, 1, 3], n_perm=50)
    assert close(pc["cross_cov"], -1.66666666666667, 1e-12) and 0 < pc["p_value"] <= 1
    with pytest.raises(ValueError):
        R.age_crime_type_cov([0.5, 0.6], [1, 2], [1, 2])


def test_feedback_loop_envelope_parity():
    # R: morie_feedback_loop_envelope(1, 1, 1, n = 40): law uniform 1/41, variance mean(((1 + 0:40)/42 - 0.5)^2)
    e = R.feedback_loop_envelope(1, 1, 1, n=40)
    assert all(close(v, 1 / 41) for v in e["law"])
    assert close(e["variance_n"], sum(((1 + j) / 42 - 0.5) ** 2 for j in range(41)) / 41)
    assert close(e["mean_n"], 0.5) and close(e["variance_bound"], 1 / 8) and e["variance_n"] < 1 / 8
    for r0, b0, a in ((1, 99, 1), (5, 3, 2), (0.5, 2, 0.7)):
        for n in (0, 1, 10, 60):
            e = R.feedback_loop_envelope(r0, b0, a, n=n, t=0.2)
            assert close(sum(e["law"]), 1)
            assert close(e["mean_n"], e["share0"])  # mean_const
            assert e["variance_n"] <= e["variance_bound"] + 1e-12  # var_le
            assert e["exit_n"] <= e["chebyshev_bound"] + 1e-12  # chebyshev
    e1 = R.feedback_loop_envelope(2, 3, 1, n=1)
    assert close(e1["variance_n"], 0.4 * 0.6 * (1 / 6) ** 2)  # step_var
    with pytest.raises(ValueError):
        R.feedback_loop_envelope(0, 1, 1)


def test_spillover_ring_bound_parity():
    # R test: path of 5 units, one treated uniformly, outer ring = distance exactly 2
    n = 5
    outer = lambda t, i: abs(t - i) == 2  # noqa: E731
    share = [sum(outer(t, i) for t in range(1, n + 1)) / n for i in range(1, n + 1)]
    gamma = [0.5, -1, 2, 0, 1.5]
    exact = sum(sum(g * outer(t, i) for g, i in zip(gamma, range(1, n + 1))) / n for t in range(1, n + 1)) / n
    r = R.spillover_ring_bound(share, gamma_max=2, gamma=gamma)
    assert close(r["expected_gap"], exact) and abs(r["expected_gap"]) <= r["bound"]
    assert close(r["bound"], 2 * sum(share) / n)
    assert R.spillover_ring_bound(share, gamma_max=2, gamma=[0] * n)["expected_gap"] == 0
    with pytest.raises(ValueError):
        R.spillover_ring_bound(share, gamma_max=1, gamma=gamma)


def test_rank_persistence_null_parity():
    from itertools import combinations

    n, k = 7, 3
    sets = list(combinations(range(1, n + 1), k))
    overlaps = [len(set(range(1, k + 1)) & set(B)) for B in sets]
    r = R.rank_persistence_null(n, k, observed=2)
    for j in range(k + 1):
        assert close(r["probability"][j], overlaps.count(j) / len(sets))
    assert close(sum(r["probability"]), 1) and close(r["mean"], k * k / n)
    assert close(r["p_value"], sum(o >= 2 for o in overlaps) / len(sets))
    big = R.rank_persistence_null(158, 16, observed=9)
    assert close(big["mean"], 256 / 158) and big["p_value"] < 1e-4
    with pytest.raises(ValueError):
        R.rank_persistence_null(5, 6)


def test_hotspot_threshold_parity():
    p = dict(A0=1, omega=1, eps=2, D=1, gamma=1, eta=0.01, z=1)
    r = R.hotspot_threshold(**p)
    B, A, rho = r["steady"]["B"], r["steady"]["A"], r["steady"]["rho"]
    assert abs(-p["omega"] * B + p["eps"] * p["D"] * rho * (p["A0"] + B)) < 1e-12  # steady_iff
    assert abs(-rho * (p["A0"] + B) + p["gamma"]) < 1e-12
    assert (round(B, 12), round(A, 12), round(rho, 12)) == (2.0, 3.0, round(1 / 3, 12))
    ks = [0.0, 0.5, 2.0, 10.0]
    tab = R.hotspot_threshold(**p, k=ks)["dispersion"]
    a = -p["omega"] + p["eps"] * p["D"] * rho
    b = p["eps"] * p["D"] * A
    c0, c1, d = -rho, 2 * p["D"] * rho / A, -A
    d1, d2 = p["eta"] * p["D"] / p["z"], p["D"]
    for row in tab:  # det_eq
        k = row["k"]
        by_hand = (a - d1 * k) * (d - d2 * k) - b * (c0 + c1 * k)
        assert abs(row["det"] - by_hand) < 1e-12
        assert abs(row["det"] - (r["P"] * k * k - r["Q"] * k + r["R"])) < 1e-12
        assert row["trace"] < 0  # tr_neg
    assert abs(r["R"] - p["omega"] * A) < 1e-12  # R_eq
    assert abs(r["Q"] - p["D"] * (3 * p["eps"] * p["D"] * rho - p["omega"] - p["eta"] * A / p["z"])) < 1e-12
    assert r["unstable"] and r["Q"] > 0 and r["Q"] ** 2 > 4 * r["P"] * r["R"]  # hotspots_above_threshold
    assert r["det_crit"] < 0 and r["growth_crit"] > 0
    tr_c = (a - d1 * r["k_crit"]) + (d - d2 * r["k_crit"])
    assert abs(r["growth_crit"] ** 2 - tr_c * r["growth_crit"] + r["det_crit"]) < 1e-10  # unstable_iff
    r0 = R.hotspot_threshold(**{**p, "eps": 0}, k=ks)  # no_hotspots_without_reinforcement
    assert not r0["unstable"] and all(row["det"] > 0 for row in r0["dispersion"])
    rb = R.hotspot_threshold(**{**p, "eps": 0.2, "eta": 0.5}, k=ks)  # no_hotspots_below_threshold
    assert rb["threshold"]["reinforcement"] <= rb["threshold"]["decay_plus_diffusion"]
    assert not rb["unstable"] and all(row["det"] > 0 for row in rb["dispersion"])
    import pytest

    with pytest.raises(ValueError, match="A0"):
        R.hotspot_threshold(0, 1, 1, 1, 1, 0.1)
    with pytest.raises(ValueError, match="eps"):
        R.hotspot_threshold(1, 1, -1, 1, 1, 0.1)


def test_dark_figure_comparable_parity():
    r = R.dark_figure_comparable(VA=1000, RA=400, RB=300, lam=0.5)
    assert r["rate_A"] == 0.4  # rate_on_A
    assert abs(r["rate_upper"] - 700 / 1300) < 1e-12  # total_rate_le
    assert r["total"] == (1300.0, 1000 + 300 / 0.2)  # total_interval
    assert r["dark_figure"] == (600.0, 600 + 300 * (1 / 0.2 - 1))  # dark_interval
    assert abs(r["width"] - 300 * (1 - 0.2) / 0.2) < 1e-12  # dark_figure_width
    for VB in (300.0, 800.0, 1500.0):
        assert r["rate_upper"] >= 700 / (1000 + VB)
        assert r["total"][0] <= 1000 + VB <= r["total"][1]
    z = R.dark_figure_comparable(VA=1000, RA=1000, RB=300, lam=1)
    assert z["width"] == 0 and z["dark_figure"] == (0.0, 0.0)
    import pytest

    with pytest.raises(ValueError, match="exceeds"):
        R.dark_figure_comparable(1000, 1200, 300)
    with pytest.raises(ValueError, match="lam"):
        R.dark_figure_comparable(1000, 400, 300, lam=0)


def test_attenuation_two_regressor_parity():
    r = R.attenuation_two_regressor(1.5, -0.5, 1, 2, 0.6, 0.5)
    assert r["theorem"] == "Research.P5.Attenuation.two_regressor_coeffs"
    assert abs((1 + 0.5) * r["b1"] + 0.6 * r["b2"] - (1.5 * 1 - 0.5 * 0.6)) < 1e-12  # normal_equations
    assert abs(0.6 * r["b1"] + 2 * r["b2"] - (1.5 * 0.6 - 0.5 * 2)) < 1e-12
    assert abs(r["dstar"] - (1.5 * 2 - 0.36)) < 1e-12
    assert abs(r["bias_b2"] - 1.5 * 0.5 * 0.6 / r["dstar"]) < 1e-12
    assert r["bias_b2_sign"] == "positive" and abs(r["reliability"] - 1 / 1.5) < 1e-12
    for args in ((0, -0.5, 1, 2, 0.6, 0.5), (1.5, -0.5, 1, 2, 0.6, 0), (1.5, -0.5, 1, 2, 0, 0.5)):
        assert R.attenuation_two_regressor(*args)["bias_b2"] == 0  # error_free_biased_iff
    assert R.attenuation_two_regressor(-1.5, -0.5, 1, 2, 0.6, 0.5)["bias_b2_sign"] == "negative"  # bias_sign
    import pytest

    with pytest.raises(ValueError, match="positive variances"):
        R.attenuation_two_regressor(1, 1, 0, 1)
    with pytest.raises(ValueError, match="s12"):
        R.attenuation_two_regressor(1, 1, 1, 1, s12=1)


def test_friedman_urn_and_ruin_parity():
    u = R.friedman_urn(9, 1, 200)
    assert u["theorem"] == "Research.P4.Friedman.meanDiff_succ" and u["mean_diff"][0] == 8
    for n in range(200):  # meanDiff_succ
        assert abs(u["mean_diff"][n + 1] - u["mean_diff"][n] * (1 - 1 / (10 + n))) < 1e-12
    assert abs(u["mean_share"][-1] - 0.5) < 0.01
    b, r = 7, 3
    f = lambda b, r: (b - r) * (b + r - 1)  # noqa: E731
    assert abs(b / (b + r) * f(b, r + 1) + r / (b + r) * f(b + 1, r) - f(b, r)) < 1e-12  # friedman_step
    p, N = 0.4, 12
    pr = R.ruin_probability(p, N)
    rho = 0.6 / 0.4
    assert pr["theorem"] == "Research.P4.Ruin.ruin_harmonic" and abs(pr["rho"] - rho) < 1e-12
    q = pr["prob"]
    assert q[0] == 1 and abs(q[N]) < 1e-12  # ruin_boundary
    for k in range(1, N):  # ruin_harmonic
        assert abs(q[k] - (p * q[k + 1] + (1 - p) * q[k - 1])) < 1e-12
    for k in range(N + 1):  # optional_stopping_identity
        assert abs(rho**k - (q[k] + (1 - q[k]) * rho**N)) < 1e-9
    assert abs(p * rho**5 + (1 - p) * rho**3 - rho**4) < 1e-12  # deMoivre_step
    sym = R.ruin_probability(0.5, N)["prob"]
    assert all(abs(sym[k] - (1 - k / N)) < 1e-12 for k in range(N + 1))  # ruin_symmetric
    assert R.ruin_probability(p, N, 3)["prob"] == [(rho**3 - rho**N) / (1 - rho**N)]
    import pytest

    with pytest.raises(ValueError, match="b0"):
        R.friedman_urn(1, 0)
    with pytest.raises(ValueError, match="0..N"):
        R.ruin_probability(0.3, 5, 6)


def test_cox_overdispersion_and_vuong_parity():
    r = R.cox_overdispersion([0.7, 0.3], [1, 5])
    m = 0.7 + 1.5
    assert r["theorem"] == "Research.P7.Cox.total_variance" and abs(r["mean"] - m) < 1e-12
    assert abs(r["intensity_var"] - (0.7 * (1 - m) ** 2 + 0.3 * (5 - m) ** 2)) < 1e-12
    assert abs(r["variance"] - (r["mean"] + r["intensity_var"])) < 1e-12  # total_variance
    assert r["variance"] > r["mean"] and not r["constant_intensity"]  # variance_ge_mean
    c0 = R.cox_overdispersion([2, 3], [4, 4])
    assert abs(c0["variance"] - c0["mean"]) < 1e-12 and c0["constant_intensity"]  # equality_iff_constant
    c1 = R.cox_overdispersion([1, 0], [2, 9])
    assert abs(c1["variance"] - c1["mean"]) < 1e-12 and c1["constant_intensity"]
    v = R.zip_vuong_boundary([0, 0, 1, 2, 0, 3])
    assert v["theorem"] == "Research.P7.Cox.llr_zero" and v["llr"] == [0.0] * 6 and v["sd"] == 0  # llr_zero
    assert v["statistic"] != v["statistic"]
    import pytest

    with pytest.raises(ValueError, match="non-negative weights"):
        R.cox_overdispersion([1, -1], [1, 2])
    with pytest.raises(ValueError, match="integers"):
        R.zip_vuong_boundary([1.5, 2])


def test_nonresponse_region_and_iv_inequality_parity():
    r = R.nonresponse_region(0.4, [0.106, 0.5, 0.9, 1])
    assert r["theorem"] == "Research.P1.Nonresponse.region_width"
    assert r["width"] == [1 - x for x in (0.106, 0.5, 0.9, 1)]  # region_width
    assert all(abs(u - lam - w) < 1e-12 for u, lam, w in zip(r["upper"], r["lower"], r["width"]))
    assert abs(r["width"][0] - 0.894) < 1e-12
    assert (
        abs(0.5 * 0.4 + 0.5 * 0 - r["lower"][1]) < 1e-12 and abs(0.5 * 0.4 + 0.5 * 1 - r["upper"][1]) < 1e-12
    )  # region_sharp
    import random

    rng = random.Random(3)
    nz, nx, ny, nu = 3, 2, 3, 4
    pu = [rng.random() for _ in range(nu)]
    pu = [v / sum(pu) for v in pu]
    px = [[[rng.random() for _ in range(nx)] for _ in range(nu)] for _ in range(nz)]
    px = [[[v / sum(row) for v in row] for row in zz] for zz in px]
    py = [[[rng.random() for _ in range(ny)] for _ in range(nu)] for _ in range(nx)]
    py = [[[v / sum(row) for v in row] for row in xx] for xx in py]
    p = [
        [[sum(pu[u] * px[z][u][x] * py[x][u][y] for u in range(nu)) for z in range(nz)] for x in range(nx)]
        for y in range(ny)
    ]
    r = R.instrumental_inequality(p)
    assert r["theorem"] == "Research.P8.IVInequality.instrumental_inequality"
    assert r["passes"] and r["max"] <= 1  # instrumental_inequality
    bad = [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]
    b = R.instrumental_inequality(bad)
    assert not b["passes"] and b["per_x"] == [2.0, 0.0]
    import pytest

    with pytest.raises(ValueError, match="sum to one"):
        R.instrumental_inequality([[[2.0, 0.0]]])
    with pytest.raises(ValueError, match="lie in"):
        R.nonresponse_region(1.2, 0.5)


def test_misclassification_parity():
    r = R.misclassified_rd(0.3, 0.1, 0.4, 0.8, 0.9)
    d1, d0 = 0.4 * 0.8 + 0.6 * 0.1, 0.4 * 0.2 + 0.6 * 0.9
    assert r["theorem"] == "Research.P2.Misclassification.binary_attenuation"
    assert abs(r["measured_rd"] - (0.8 + 0.9 - 1) * 0.2 * 0.4 * 0.6 / (d1 * d0)) < 1e-12  # binary_attenuation
    assert abs(r["measured_rd"] - r["factor"] * r["true_rd"]) < 1e-12
    assert r["sign_kept"] and abs(r["measured_rd"]) < abs(r["true_rd"])  # binary_sign
    assert abs(R.misclassified_rd(0.3, 0.1, 0.4, 1, 1)["measured_rd"] - 0.2) < 1e-12
    assert abs(R.misclassified_rd(0.3, 0.1, 0.4, 0.5, 0.5)["measured_rd"]) < 1e-12
    w = R.misclassified_contrast([0, 0.5, 1], ["high", "low", "high"])
    assert w["theorem"] == "Research.P2.Misclassification.ordinal_witness"
    assert w["true_contrast"] == 1 and w["measured_contrast"] == 0  # ordinal_witness
    rv = R.misclassified_contrast([0, 0.5, 1], ["high", "high", "low"])
    assert rv["measured_contrast"] == 0.25 - 1 and rv["reversed"]  # ordinal_reversal
    import pytest

    with pytest.raises(ValueError, match="prevalence"):
        R.misclassified_rd(0.3, 0.1, 1, 0.8, 0.9)
    with pytest.raises(ValueError, match="both appear"):
        R.misclassified_contrast([0, 1], ["a", "b"])


def test_amplification_and_design_effect_parity():
    g = R.sensitivity_amplification(Lambda=[2, 3, 5], Delta=4)
    assert g["theorem"] == "Research.P11.Amplification.level_curve"
    assert all(abs(gg - (lam * 4 + 1) / (lam + 4)) < 1e-12 for gg, lam in zip(g["Gamma"], (2, 3, 5)))
    assert g["Gamma"] == R.sensitivity_amplification(Lambda=4, Delta=[2, 3, 5])["Gamma"]  # symmetric
    assert all(1 <= gg <= min(lam, 4) for gg, lam in zip(g["Gamma"], (2, 3, 5)))  # ge_one, le_min
    assert g["Gamma"] == sorted(g["Gamma"])  # mono_left
    cv = R.sensitivity_amplification(Gamma=2, Lambda=[3, 4, 10])
    assert all(abs(d - (lam * 2 - 1) / (lam - 2)) < 1e-12 for d, lam in zip(cv["Delta"], (3, 4, 10)))  # level_curve
    back = R.sensitivity_amplification(Lambda=cv["Lambda"], Delta=cv["Delta"])["Gamma"]
    assert all(abs(v - 2) < 1e-12 for v in back)
    assert abs(R.sensitivity_amplification(Gamma=2, Delta=5)["Lambda"][0] - 3) < 1e-12
    d = R.design_effect_direction(0.1, 20)
    assert d["theorem"] == "Research.P13.Design.quad_form"
    assert abs(d["deff_cluster_level"] - 2.9) < 1e-12 and abs(d["deff_within_balanced"] - 0.9) < 1e-12
    assert abs(d["icc_floor"] + 1 / 19) < 1e-12  # icc_lower_bound
    xc = [[2] * 4, [-1] * 4, [3] * 4]
    assert abs(R.design_effect_direction(0.1, 4, xc)["ratio"] - 1.3) < 1e-12  # cluster_level
    xb = [[1, -1], [2, -2], [0.5, -0.5]]
    assert abs(R.design_effect_direction(0.1, 2, xb)["ratio"] - 0.9) < 1e-12  # within_balanced
    import random

    rng = random.Random(1)
    x = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(3)]
    dq = R.design_effect_direction(0.3, 4, x)
    by_hand = sum(v * (0.7 * v + 0.3 * sum(row)) for row in x for v in row)  # quad_form
    assert abs(dq["quad"] - by_hand) < 1e-12
    ind = [[1] * 4, [0] * 4, [0] * 4]
    assert abs(R.design_effect_direction(-1 / 3, 4, ind)["quad"]) < 1e-12  # quad_indicator at the floor
    import pytest

    with pytest.raises(ValueError, match="exactly two"):
        R.sensitivity_amplification(Lambda=2)
    with pytest.raises(ValueError, match="must exceed"):
        R.sensitivity_amplification(Gamma=2, Lambda=2)
    with pytest.raises(ValueError, match="lie in"):
        R.design_effect_direction(-0.6, 3)
    with pytest.raises(ValueError, match="M columns"):
        R.design_effect_direction(0.1, 3, [[1, 1]])


def test_basu_contraction_inspection_parity():
    t, pi = [1, 2, 3, 100], [0.5, 0.5, 0.5, 0.01]
    r = R.ht_elephant(t, pi, bound=1e4)
    assert r["theorem"] == "Research.P3.Basu.ht_variance" and r["mean"] == 106  # ht_mean
    assert all(abs(a - b * b * (1 - c) / c) < 1e-9 for a, b, c in zip(r["terms"], t, pi))  # ht_variance
    assert abs(r["variance"] - sum(r["terms"])) < 1e-9
    assert r["elephant"] == 3 and abs(r["pi_elephant"] - 100**2 / (2 * (1e4 + 100**2))) < 1e-12
    assert r["term_elephant"] > 1e4 and r["pi_elephant"] < 1  # elephant_unbounded
    assert sum(r["terms"][:3]) <= sum(v * v / q for v, q in zip(t[:3], pi[:3]))  # ratio_bounded
    S = [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    c = R.attractiveness_contraction(S, 0.2, 0.3, [5, 1, 0], 6)
    assert c["theorem"] == "Research.P4.Contraction.decay"
    assert all(a <= b + 1e-12 for a, b in zip(c["sup_norm"], c["sup_bound"]))  # decay
    assert c["doubly_stochastic"] and all(abs(v - 0.8**k * 6) < 1e-12 for k, v in enumerate(c["total"]))  # update_total
    assert abs(c["feedback_threshold"] - 0.25) < 1e-12  # feedback_gain
    assert 0.8 * (1 + 0.3) > 1 and not 0.8 * (1 + 0.2) > 1
    c2 = R.attractiveness_contraction([[1, 0], [0.5, 0.5]], 0, 1, [1, 3], 2)
    assert not c2["doubly_stochastic"] and c2["sup_norm"][1] == 2 and c2["feedback_threshold"] == 0
    g = R.inspection_game(1, [0.5, 0.3, 0.2], 2000, 0.01)
    assert g["theorem"] == "Research.P8.Inspection.conserved" and abs(g["dV_dt"]) < 1e-15  # conserved
    assert abs(sum(g["field"])) < 1e-15  # simplex_invariant / payoff_zero_sum
    assert g["invariant_drift"] < 1e-8
    assert all(abs(sum(row) - 1) < 1e-10 for row in g["trajectory"])
    assert max(abs(row[0] - 0.5) for row in g["trajectory"]) > 0.1
    c3 = R.inspection_game(2, [1 / 3, 1 / 3, 1 / 3], 10)
    assert c3["field"] == [0, 0, 0] and all(
        abs(v - 1 / 3) < 1e-12 for v in c3["trajectory"][-1]
    )  # interior_fixed_point
    import pytest

    with pytest.raises(ValueError, match="unit row sums"):
        R.attractiveness_contraction([[2, 0], [0, 1]], 0.2, 0.3, [1, 1])
    with pytest.raises(ValueError, match="three shares"):
        R.inspection_game(1, [0.5, 0.5])
    with pytest.raises(ValueError, match="bound"):
        R.ht_elephant(t, pi, bound=0)


def test_scan2_zero_inflation_gamma_blocking_adjustment_parity():
    import math

    import pytest

    r = R.nb_matches_zip(0.3, 2)
    assert r["theorem"] == "Research.P7.ZeroInflation.nb_matches_zip"
    assert abs(r["mean"] - 1.4) < 1e-12 and abs(r["zip_zero"] - (0.3 + 0.7 * math.exp(-2))) < 1e-12
    assert math.exp(-r["mean"]) < r["zip_zero"] < 1  # zip_zero_gt_poisson, zip_zero_lt_one
    assert abs(r["nb_zero"] - r["zip_zero"]) < 1e-9  # nb_matches_zip
    with pytest.raises(ValueError, match="pi"):
        R.nb_matches_zip(1, 2)
    g = R.gamma_bound(70, 100, [1, 1.5, 2, 7 / 3, 3])
    assert g["theorem"] == "Research.P5.Gamma.gamma_star_kills" and abs(g["gamma_star"] - 70 / 30) < 1e-12
    wm = g["worst"]["worst_mean"]
    assert wm == sorted(wm) and abs(wm[3] - 70) < 1e-12  # worst_mean_mono, worst_mean_at_gamma_star
    assert g["worst"]["survives"] == [True, True, True, False, False]
    with pytest.raises(ValueError, match="T must"):
        R.gamma_bound(100, 100)
    b = R.blocking_crossover(0.5, 1, 10, rho=[0.02, 0.3, 0.9])
    assert b["theorem"] == "Research.P3.Blocking.paired_better_iff"
    assert b["c_paired"] > b["c_unpaired"] and b["crossover"] > 0  # crossover_pos
    assert abs(b["c_paired"] - 1.8331129) < 1e-5 and abs(b["c_unpaired"] - 1.7340636) < 1e-5
    assert b["paired_better"] == [x > b["crossover"] for x in (0.02, 0.3, 0.9)]  # paired_better_iff
    a = R.adjustment_variances(1, 0, 1, 1, 0.8)
    assert a["theorem"] == "Research.P3.Adjustment.adjusted_minus_unadjusted" and a["adjustment_hurts"]
    assert abs(a["adjusted_minus_unadjusted"] - (3 * 0.8 - 2) / 0.2) < 1e-12  # adjusted_worse
    assert not R.adjustment_variances(1, 0, 1, 1, 0.6)["adjustment_hurts"]
    for p in (0.3, 0.5, 0.7):
        s = R.adjustment_variances(1.2, -0.4, 0.5, 2, p)
        assert s["lin"] <= s["unadjusted"] and s["lin"] <= s["adjusted"]  # lin_le_*
    s = R.adjustment_variances(1.2, -0.4, 0.5, 2, 0.5)
    assert s["adjusted"] <= s["unadjusted"]  # adjusted_le_unadjusted_balanced


def test_scan2_budget_outcome_spatial_bounds_parity():
    import math

    import pytest

    o = R.budget_optimum(math.sqrt, lambda s: 2 * math.log1p(s), 10)
    assert o["theorem"] == "Research.P8.Budget.interior_optimum" and o["interior"]
    assert o["value"] > o["corner_values"]["stick"] and o["value"] > o["corner_values"]["carrot"]
    f = o["grid"]["objective"]
    assert all(f[i] >= (f[i - 1] + f[i + 1]) / 2 - 1e-12 for i in range(1, len(f) - 1))  # concave_sum
    assert not R.budget_optimum(lambda t: 0.01 * t, lambda s: s, 1)["interior"]
    t = R.outcome_test({"A": {"q": [0.3, 0.9], "w": [1, 1], "t": 0.5}, "B": {"q": [0.9], "w": [1], "t": 0.2}})
    assert t["theorem"] == "Research.P2.OutcomeTest.infra_marginality"
    assert (
        t["A"]["hit_rate"] == t["B"]["hit_rate"] == 0.9 and t["A"]["stop_rate"] == 0.5 and t["B"]["stop_rate"] == 1
    )  # infra_marginality
    t2 = R.outcome_test({"A": {"q": [0.3, 0.9], "w": [1, 1], "t": 0.5}, "B": {"q": [0.6, 0.9], "w": [1, 1], "t": 0.5}})
    assert t2["A"]["hit_rate"] != t2["B"]["hit_rate"]  # equal_thresholds_unequal_hits
    q = [(i + 0.5) / 1000 for i in range(1000)]
    u = R.outcome_test({"U": {"q": q, "w": [1] * 1000, "t": 0.4}})
    assert abs(u["U"]["hit_rate"] - 0.7) < 1e-3  # uniform_hit_rate
    with pytest.raises(ValueError, match="q in"):
        R.outcome_test({"A": {"q": [2], "w": [1], "t": 0.1}})
    s = R.spatial_omission(0.4, [1, 2], [[1, 0.5], [0.5, 1]], 0.05, 50)
    assert s["theorem"] == "Research.P3.Spatial.lag_regressor_correlated"
    assert (
        abs(s["lag_covariance"] - 0.4 / 0.84) < 1e-12
        and abs(s["var_ols"] - 7 / 25) < 1e-12
        and abs(s["var_gls"] - 0.25) < 1e-12
    )
    assert s["gls_more_precise"] and abs(s["fwer"] - (1 - 0.95**50)) < 1e-12  # gls_more_precise, fwer
    nb = R.iv_natural_bounds(0.8, 0.3, 0.6, 0.4)
    assert nb["theorem"] == "Research.P8.NaturalBounds.width" and abs(nb["width"] - 0.5) < 1e-12  # width
    ate = lambda q1, q0: (0.6 * 0.8 + q1 * 0.2) - (0.4 * 0.7 + q0 * 0.3)  # noqa: E731
    assert abs(ate(0, 1) - nb["lower"]) < 1e-12 and abs(ate(1, 0) - nb["upper"]) < 1e-12  # bounds_attained
    assert (
        R.iv_natural_bounds(1, 0, 0.6, 0.4)["width"] == 0
        and abs(R.iv_natural_bounds(0.5, 0.5, 0.6, 0.4)["width"] - 1) < 1e-12
    )
    d = R.design_sensitivity(0.7)
    assert d["theorem"] == "Research.P5.DesignSensitivity.design_sensitivity_exists_unique"
    assert abs(d["gamma_tilde"] - 7 / 3) < 1e-8 and d["limiting_power"]["power"] == [
        1.0,
        1.0,
        0.0,
    ]  # eventually_reject/accept
    with pytest.raises(ValueError, match="no effect"):
        R.design_sensitivity(0.4)


def test_scan2_moran_exposure_thinning_chernoff_walk_parity():
    import math
    import random

    import pytest

    W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    m = R.moran_trend(W, [1, 2, 3, 4])
    assert m["theorem"] == "Research.P7.MoranTrend.residual_moran_ratio"
    assert abs(m["I"] - 1 / 3) < 1e-12 and abs(m["null_expectation"] + 1 / 3) < 1e-12  # trend_witness
    n = 12
    Wn = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]
    X = [[1.0, float(i + 1)] for i in range(n)]
    rng = random.Random(5)
    y = [2 + 0.5 * (i + 1) + rng.gauss(0, 1) for i in range(n)]
    mr = R.moran_trend(Wn, y, X)
    assert "I_residual" in mr and abs(mr["residual_expectation"]) < 1  # residual_moran_ratio form
    with pytest.raises(ValueError, match="n x n"):
        R.moran_trend(W, [1, 2, 3])
    e = R.exposure_positivity(0.5, [1, 5, 10, 30])
    assert e["theorem"] == "Research.P3.Exposure.exposure_prob_tendsto_zero"
    assert e["prob_all_treated"] == [0.5**d for d in (1, 5, 10, 30)] and e["ht_weight"][-1] == 2.0**30
    stats = [rng.gauss(0, 1) for _ in range(200)]
    pv = R.permutation_valid(stats, [0.01, 0.05, 0.1, 0.5])
    assert pv["theorem"] == "Research.P3.Exposure.permutation_valid" and all(pv["levels"]["valid"])  # permutation_valid
    assert R.permutation_valid([1, 1, 1, 2], [0.5])["levels"]["rejection_share"] == [0.25]
    t = R.thinning_identification(5, 0.3)
    assert (
        t["theorem"] == "Research.P1.Thinning.not_identified" and t["recorded_rate"] == 1.5 and t["pgf_check"]
    )  # thinned_pgf
    assert t["alternative"]["lam"] * t["alternative"]["gamma"] == 1.5 and t["alternative"]["lam"] != 5  # not_identified
    s, u = 0.4, 0.7
    assert (
        abs(math.exp(5 * ((0.3 * s + 0.7 * u) - 1)) - math.exp(1.5 * (s - 1)) * math.exp(3.5 * (u - 1))) < 1e-12
    )  # joint_pgf_factorises
    c = R.poisson_chernoff(20, [0.25, 0.5, 1])
    assert c["theorem"] == "Research.P7.Chernoff.poisson_chernoff"
    assert all(r > 0 for r in c["rate"]) and all(
        a <= b for a, b in zip(c["exact"], c["bound"])
    )  # rate_pos, poisson_chernoff
    adj = [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]]
    w = R.random_walk_exposure(adj)
    assert w["theorem"] == "Research.P3.RandomWalk.stationary" and w["undirected"] and w["stationary"]  # stationary
    assert abs(sum(w["pi"]) - 1) < 1e-12 and w["pi"] == [2 / 8, 3 / 8, 2 / 8, 1 / 8]  # stationary_sum_one
    dd = R.random_walk_exposure([[0, 1, 1], [0, 0, 1], [1, 0, 0]])
    assert not dd["undirected"] and not dd["stationary"]  # directed_fails
    with pytest.raises(ValueError, match="out-edge"):
        R.random_walk_exposure([[0, 0], [0, 0]])


def test_scan2_extremal_tail_calibration_bellman_budget_cusum_parity():
    import math
    import random

    import pytest

    x = R.extremal_index_clusters(4, 2, [10, 100, 100000])
    assert x["theorem"] == "Research.P10.ExtremalIndex.extremal_index" and x["theta"] == 0.25
    assert abs(x["limit_cluster"] ** 4 - x["limit_iid"]) < 1e-12  # extremal_index
    assert abs(x["table"]["cluster"][2] - math.exp(-0.5)) < 1e-4 and abs(x["table"]["iid"][2] - math.exp(-2)) < 1e-4
    tr = R.tail_ratio(
        [5, 10, 20, 40], q=0.7, alpha=1.5, r=3, counts=[1] * 50 + [3] * 30 + [8] * 15 + [20, 25, 30, 40, 60]
    )
    assert tr["theorem"] == "Research.P7.TailShape.nb_pmf_ratio_tendsto_zero"
    assert tr["geometric"] == [0.7**k for k in (5, 10, 20, 40)] and all(abs(p - 2**-1.5) < 1e-15 for p in tr["power"])
    assert (
        all(a <= b for a, b in zip(tr["nb_pmf_ratio"], tr["nb_bound"])) and tr["nb_pmf_ratio"][-1] < 1e-3
    )  # nb_pmf_ratio_le, tendsto
    assert (
        abs(
            tr["empirical"][0]
            - (
                sum(1 for v in [1] * 50 + [3] * 30 + [8] * 15 + [20, 25, 30, 40, 60] if v > 10)
                / sum(1 for v in [1] * 50 + [3] * 30 + [8] * 15 + [20, 25, 30, 40, 60] if v > 5)
            )
        )
        < 1e-12
    )
    cf = R.calibration_flip(0.05, 0.2, [0.3, 0.5, 0.8])
    assert cf["theorem"] == "Research.P5.DataProcessing.calibration_incompatible"
    assert all(abs(0.05 + 0.9 * a - r) < 1e-12 for a, r in zip(cf["true_A"], cf["r"]))  # calibrated_to_D_true_rate
    assert (
        cf["incompatible"] == [True, False, True]
        and cf["true_A"][0] != cf["true_B"][0]
        and cf["recorded_A"][2] != cf["recorded_B"][2]
    )
    dp = R.bayes_risk_dpi([[0.3, 0.1, 0.05], [0.1, 0.2, 0.25]], [0, 1], [1, 1, 2])
    assert (
        dp["theorem"] == "Research.P5.DataProcessing.bayes_risk_data_processing"
        and dp["dpi_holds"]
        and dp["risk_D"] <= dp["risk_R"]
    )
    assert (
        abs(R.bayes_risk_dpi([[0.3, 0.1, 0.05], [0.1, 0.2, 0.25]], [0, 1], [1, 2, 3])["risk_R"] - dp["risk_D"]) < 1e-12
    )
    r = [[1, 0], [0.5, 0.5], [1, 1]]
    p = [[[0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]]]
    b = R.bellman_value(r, p, 0.9)
    assert (
        b["theorem"] == "Research.P4.Bellman.bellman_contracting"
        and b["converged"]
        and b["contraction_ratio"] <= 0.9 + 1e-6
    )  # bellman_lipschitz
    assert (
        abs(b["value"][1] - 5) < 1e-6
        and abs(b["value"][2] - 10) < 1e-6
        and abs(b["value"][0] - 9) < 1e-6
        and b["policy"][0] == 1
    )
    q = [[r[s][a] + 0.9 * sum(p[s][a][t] * b["value"][t] for t in range(3)) for a in range(2)] for s in range(3)]
    assert all(abs(max(row) - v) < 1e-6 for row, v in zip(q, b["value"]))  # fixed point
    me = R.myopic_example([0.5, 2 / 3, 0.9])
    assert me["theorem"] == "Research.P4.Bellman.myopic_suboptimal" and me["patient_better"] == [
        False,
        False,
        True,
    ]  # myopic_suboptimal
    br = R.budget_rule([1, 1, 1], [0.5, 0.2, -0.1], 1.5)
    assert br["theorem"] == "Research.P5.BudgetRule.threshold_optimal" and br["rule"] == [1.0, 0.5, 0.0]
    assert (
        abs(br["value"] - 0.6) < 1e-12 and br["stochastic_needed"] and br["best_deterministic"] == 0.5
    )  # stochastic_needed
    rng = random.Random(9)
    for _ in range(200):  # threshold_optimal
        d = [rng.random() for _ in range(3)]
        if sum(d) > 1.5:
            d = [v * 1.5 / sum(d) for v in d]
        assert sum(a * b for a, b in zip(d, (0.5, 0.2, -0.1))) <= br["value"] + 1e-12
    rl = R.run_length_bound(0.01)
    assert rl["theorem"] == "Research.P10.Cusum.mean_run_length_ge" and rl["J"] == 100 and rl["bound"] == 50
    assert (
        abs(rl["partial_sum"] - sum(1 - n * 0.01 for n in range(100))) < 1e-12 and rl["partial_sum"] >= rl["bound"]
    )  # partial_sum_ge_half
    N = []
    for _ in range(4000):  # stop_prob_le
        k = 0
        while rng.random() >= 0.05:
            k += 1
        N.append(k)
    for n in (0, 3, 10):
        assert sum(1 for v in N if v <= n) / 4000 <= (n + 1) * 0.05 + 0.03
    with pytest.raises(ValueError, match="must not exceed 1"):
        R.run_length_bound(0.5, J=3)
