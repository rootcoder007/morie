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
