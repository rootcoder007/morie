"""Tests for the Vee-sourced batch (strec, edgrn, inlasm, bhltmsm)
and the sobolI correction."""
import importlib
import math

import pytest


def M(name):
    return importlib.import_module("morie.fn." + name)


# ----------------------------------------------------------------- strec
SESS = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
I2 = [[1.0, 0.0], [0.0, 1.0]]


def test_strec_trilinear_is_the_hadamard_form():
    st = M("strec")
    assert st.trilinear([1.0, 2.0], [3.0, 4.0],
                        [5.0, 6.0]) == pytest.approx(63.0)
    with pytest.raises(ValueError):
        st.trilinear([1.0], [1.0, 2.0], [1.0])


def test_strec_product_needs_both_memories_where_a_sum_does_not():
    st = M("strec")
    h_s, h_t = [10.0, 1.0], [0.01, 1.0]
    items = [[1.0, 0.0], [0.0, 1.0]]
    tri = [st.trilinear(h_s, h_t, v) for v in items]
    add = [sum((h_s[a] + h_t[a]) * v[a] for a in range(2))
           for v in items]
    assert add[0] > add[1]
    assert tri[1] > tri[0]


def test_strec_ms_is_the_mean_and_mt_the_last_click():
    st = M("strec")
    r = st.session_average(SESS)
    assert r["m_s"] == [0.5, 0.5]
    assert r["m_t"] == [1.0, 0.0]
    with pytest.raises(ValueError):
        st.session_average([])


def test_strec_attention_is_not_normalised_and_sees_the_last_click():
    st = M("strec")
    W = [[1.0, 0.0], [0.0, 1.0]]
    a = st.attention_weights(SESS, W, W, W, [1.0, 1.0], [0.0, 0.0])
    alt = st.attention_weights([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
                                [0.0, 1.0]], W, W, W, [1.0, 1.0],
                               [0.0, 0.0])
    assert abs(a["sum_alpha"] - 1.0) > 0.1
    assert abs(a["alpha"][0] - alt["alpha"][0]) > 1e-6


def test_strec_stamp_and_stmp_differ():
    st = M("strec")
    W = [[1.0, 0.0], [0.0, 1.0]]
    att = st.attention_weights(SESS, W, W, W, [1.0, 1.0], [0.0, 0.0])
    stmp = st.stamp_scores(SESS, I2, I2, I2)
    stamp = st.stamp_scores(SESS, I2, I2, I2, attention=att)
    assert stmp["model"] == "STMP" and stamp["model"] == "STAMP"
    assert stmp["score"] != stamp["score"]


def test_strec_cross_entropy_prefers_the_truth():
    st = M("strec")
    assert st.cross_entropy([0.9, 0.1], 0) < st.cross_entropy(
        [0.9, 0.1], 1)
    with pytest.raises(ValueError):
        st.cross_entropy([0.9, 0.1], 5)


# ----------------------------------------------------------------- edgrn
def test_edgrn_dispersion_zero_is_poisson():
    eg = M("edgrn")
    r = eg.nb_variance(100.0, 0.0)
    assert r["variance"] == pytest.approx(100.0)
    assert r["biological"] == pytest.approx(0.0)
    assert eg.nb_variance(100.0, 0.04)["bcv"] == pytest.approx(0.2)


def test_edgrn_tmm_recovers_the_non_de_ratio_exactly():
    eg = M("edgrn")
    ref = [100.0] * 100
    smp = [200.0] * 25 + [100.0] * 75
    r = eg.tmm_factor(smp, ref)
    assert r["factor"] == pytest.approx(0.8, abs=1e-9)
    eff = eg.effective_library_size(sum(smp), r["factor"])
    assert eff["effective"] == pytest.approx(10000.0)


def test_edgrn_tmm_breaks_past_the_documented_range():
    eg = M("edgrn")
    ref = [100.0] * 100
    bad = eg.tmm_factor([200.0] * 60 + [100.0] * 40, ref)
    assert abs(bad["factor"] - 10000.0 / 16000.0) > 0.4


def test_edgrn_tmm_needs_a_gene_positive_in_both():
    eg = M("edgrn")
    with pytest.raises(ValueError):
        eg.tmm_factor([0.0, 0.0], [1.0, 1.0])


def test_edgrn_moderation_shrinks_toward_the_common_value():
    eg = M("edgrn")
    raw = [0.001, 0.5, 0.05]
    r = eg.moderate_dispersion(raw, prior_df=10.0, df_residual=1.0)
    for i in range(3):
        assert abs(r["dispersion"][i] - r["common"]) < abs(
            raw[i] - r["common"])


def test_edgrn_exact_test_separates_shift_from_no_shift():
    eg = M("edgrn")
    assert eg.exact_test(100.0, 10.0, 1000.0, 1000.0,
                         0.01)["p_value"] < 0.01
    assert eg.exact_test(50.0, 50.0, 1000.0, 1000.0,
                         0.01)["p_value"] > 0.5


def test_edgrn_ql_f_test_matches_the_t_identity_and_beats_the_lrt():
    eg = M("edgrn")
    r = eg.ql_f_test(4.0, 1, 1.0, 10.0)
    assert r["p_value"] == pytest.approx(0.073388, abs=1e-5)
    conservative = eg.ql_f_test(10.0, 1, 1.0, 6.0)
    assert conservative["p_value"] > conservative["lrt_p_value"]


# ---------------------------------------------------------------- inlasm
def test_inlasm_gaussian_likelihood_is_exact():
    il = M("inlasm")
    tau, y, m0, q0 = 2.0, 3.0, 1.0, 0.5
    r = il.gaussian_approximation(
        lambda x: -0.5 * tau * (y - x) ** 2,
        lambda x: tau * (y - x), lambda x: -tau, m0, q0)
    assert r["mode"] == pytest.approx((q0 * m0 + tau * y) / (q0 + tau),
                                      abs=1e-12)
    assert r["precision"] == pytest.approx(q0 + tau, abs=1e-12)


def test_inlasm_non_concave_objective_refused():
    il = M("inlasm")
    with pytest.raises(ValueError):
        il.gaussian_approximation(lambda x: x ** 3,
                                  lambda x: 3 * x * x,
                                  lambda x: 6 * x, 0.0, 0.5, x0=5.0)


def test_inlasm_skewness_is_zero_only_for_a_gaussian():
    il = M("inlasm")
    assert il.skewness_correction(0.0, 2.5)["gaussian_adequate"]
    assert not il.skewness_correction(0.7, 2.5)["gaussian_adequate"]


def test_inlasm_design_is_small_and_finite():
    il = M("inlasm")
    r = il.hyperparameter_design([0.0, 0.0], [1.0, 4.0])
    assert r["n_points"] == 5
    with pytest.raises(ValueError):
        il.hyperparameter_design([0.0] * 8, [1.0] * 8)


def test_inlasm_outer_sum_weights_the_marginals():
    il = M("inlasm")
    grid = [(-6.0 + 12.0 * i / 400.0) for i in range(401)]

    def gauss(mu):
        return [math.exp(-0.5 * (x - mu) ** 2) / math.sqrt(2 * math.pi)
                for x in grid]

    even = il.integrate_marginals([gauss(-1.0), gauss(1.0)],
                                  [0.0, 0.0], grid)
    tilt = il.integrate_marginals([gauss(-1.0), gauss(1.0)],
                                  [math.log(3.0), 0.0], grid)
    assert even["mean"] == pytest.approx(0.0, abs=1e-9)
    assert tilt["mean"] == pytest.approx(-0.5, abs=1e-6)
    assert tilt["theta_weights"][0] == pytest.approx(0.75, abs=1e-12)


# --------------------------------------------------------------- bhltmsm
def test_bhltmsm_cumulative_counts_the_periods():
    bh = M("bhltmsm")
    r = bh.cumulative_episodes([["none", "outpatient", "outpatient"]])
    assert r["cumulative"][0] == [1.0, 2.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        bh.cumulative_episodes([["detox"]])


def test_bhltmsm_weights_stabilise_and_reject_zero_propensity():
    bh = M("bhltmsm")
    same = bh.treatment_weights([["none", "none"]], [[0.5, 0.5]],
                                stabilise=True, marginal=[[0.5, 0.5]])
    assert same["weights"][0] == pytest.approx(1.0)
    plain = bh.treatment_weights([["none", "none"]], [[0.5, 0.5]],
                                 stabilise=False)
    assert plain["weights"][0] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        bh.treatment_weights([["none"]], [[0.0]], stabilise=False)


def test_bhltmsm_diagnostics_catch_a_dominating_weight():
    bh = M("bhltmsm")
    r = bh.weight_diagnostics([1.0] * 99 + [100.0])
    assert r["effective_n"] < 25.0
    assert not r["mean_near_one"]


def test_bhltmsm_iptw_fixes_a_sign_reversal():
    bh = M("bhltmsm")
    rng = M("_array_core").random.default_rng(5)
    cum, y, w = [], [], []
    for _ in range(400):
        sick = 1.0 if float(rng.uniform()) < 0.5 else 0.0
        p = 0.9 if sick else 0.2
        treated = 1.0 if float(rng.uniform()) < p else 0.0
        pr = p if treated else 1.0 - p
        cum.append([treated])
        y.append(5.0 * sick - 2.0 * treated
                 + 0.2 * (float(rng.uniform()) - 0.5))
        w.append((0.55 if treated else 0.45) / pr)
    assert bh.fit_msm(y, cum)["estimate"] > 0.0
    assert abs(bh.fit_msm(y, cum, w,
                          states=("treated",))["estimate"] + 2.0) < 0.6


def test_bhltmsm_detects_the_feedback_condition():
    bh = M("bhltmsm")
    r = bh.confounding_check([[1.0, 1.0], [0.0, 0.0]] * 20,
                             [[1.0, 1.0], [0.0, 0.0]] * 20)
    assert r["is_treatment_confounder_feedback"]


# ---------------------------------------------------------------- sobolI
def test_sobolI_ishigami_closed_form():
    sb = M("sobolI")
    ex = sb.ishigami_exact()
    assert ex["S"][0] == pytest.approx(0.3139, abs=5e-4)
    assert ex["S"][1] == pytest.approx(0.4424, abs=5e-4)
    assert ex["S"][2] == 0.0
    assert ex["ST"][2] == pytest.approx(0.2437, abs=5e-4)


def test_sobolI_estimates_match_the_closed_form():
    sb = M("sobolI")
    ex = sb.ishigami_exact()
    pi = math.pi

    def ppf(u):
        return -pi + 2.0 * pi * u

    r = sb.sobol_indices(sb.ishigami, [ppf] * 3, N=4096, d=3)
    for i in range(3):
        assert abs(r["S"][i] - ex["S"][i]) < 0.03
        assert abs(r["ST"][i] - ex["ST"][i]) < 0.04
    assert r["model_runs"] == 4096 * 5


def test_sobolI_first_order_and_total_are_different_questions():
    sb = M("sobolI")
    pi = math.pi
    r = sb.sobol_indices(sb.ishigami,
                         [lambda u: -pi + 2.0 * pi * u] * 3,
                         N=4096, d=3)
    assert abs(r["S"][2]) < 0.03
    assert r["ST"][2] > 0.2
    assert not r["additive"]


def test_sobolI_designs_are_distinct_and_named_correctly():
    sb = M("sobolI")
    hal = sb.sample_matrices(16, 2, design="halton")
    sob = sb.sample_matrices(16, 2, design="sobol")
    assert hal["A"] != sob["A"]
    assert hal["design"] == "halton" and sob["design"] == "sobol"
    with pytest.raises(ValueError):
        sb.sample_matrices(16, 2, design="latin")


def test_sobolI_constant_model_refused():
    sb = M("sobolI")
    with pytest.raises(ValueError):
        sb.sobol_indices(lambda x: 1.0, None, N=32, d=2)
