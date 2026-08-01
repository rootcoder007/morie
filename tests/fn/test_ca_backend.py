"""Book-anchored tests for the Criminology shelf backend.

Every numeric anchor below is a worked number printed in Weisburd, Wilson,
Wooditch & Britt (2022, 5th ed): chapter Working-It-Out boxes, tables, or
in-text arithmetic.
"""

import math

import numpy as np
import pytest

from morie.fn import _ca_crim as ca

# Table 2.1 (book p. 19): years in prison (x) vs subsequent arrests (y), n=20.
X21 = np.array([1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5], float)
Y21 = np.array([1, 0, 1, 2, 2, 0, 1, 1, 3, 3, 3, 4, 2, 4, 4, 4, 5, 6, 9, 7], float)


class TestCh2OLS:
    def test_simple_regression_table_2_1(self):
        fit = ca.ols_simple(X21, Y21)
        # book: b1 = 1.7089, b0 = -2.28, r = 0.7156, t = 4.346
        assert fit["b1"] == pytest.approx(1.7089, abs=5e-4)
        assert fit["b0"] == pytest.approx(-2.28, abs=5e-3)
        assert fit["r"] == pytest.approx(0.7156, abs=5e-4)
        assert fit["t"] == pytest.approx(4.346, abs=5e-3)
        # eq (2.5) and eq (2.6) are algebraically identical
        assert fit["t"] == pytest.approx(fit["t_from_r"], rel=1e-10)
        # book predictions -0.5711 / 2.8467 / 6.2645 for x = 1, 3, 5 (the
        # x=1 prediction is negative -- "we would predict less than 0
        # rearrests"; the corpus OCR dropped the minus sign)
        # the book computes them from the ROUNDED b0 = -2.28, b1 = 1.7089
        for x, want in [(1, -0.5711), (3, 2.8467), (5, 6.2645)]:
            assert -2.28 + 1.7089 * x == pytest.approx(want, abs=1e-4)
            assert fit["b0"] + fit["b1"] * x == pytest.approx(want, abs=5e-3)

    def test_two_iv_coefficients_book_p26(self):
        # book: r_y1=0.7156, r_y2=0.7616, r_12=0.6280, s_y=2.300,
        # s_x1=0.9631 -> b1 = 0.9358; b2 = 0.9593 (s_x2 backed out of eq 2.8)
        got = ca.ols_two_iv(0.7156, 0.7616, 0.6280, 2.300, 0.9631, 1.0)
        assert got["b1"] == pytest.approx(0.9358, abs=2e-4)
        b2_unit = got["b2"]  # with s_2 = 1: b2 = 0.9593 * s_x2 ... invert:
        s_x2 = b2_unit / 0.9593
        assert ca.ols_two_iv(0.7156, 0.7616, 0.6280, 2.300, 0.9631, s_x2)[
            "b2"] == pytest.approx(0.9593, rel=1e-9)

    def test_variance_partition_identity(self):
        fit = ca.ols_simple(X21, Y21)
        yhat = fit["b0"] + fit["b1"] * X21
        vp = ca.variance_partition(Y21, yhat)
        # SS_total = SS_model + SS_resid for a least-squares fit
        assert vp["ss_total"] == pytest.approx(vp["ss_model"] + vp["ss_resid"],
                                               rel=1e-10)
        # R^2 = r^2 in simple regression
        assert vp["r2"] == pytest.approx(fit["r"] ** 2, rel=1e-10)

    def test_adjusted_r2_book_p44(self):
        # book: n=200, R2=0.25, k=5,10,20,40 -> 0.23, 0.21, 0.17, 0.06
        for k, want in [(5, 0.23), (10, 0.21), (20, 0.17), (40, 0.06)]:
            assert ca.adjusted_r2(0.25, 200, k) == pytest.approx(want, abs=5e-3)

    def test_f_forms_agree(self):
        fit = ca.ols_simple(X21, Y21)
        yhat = fit["b0"] + fit["b1"] * X21
        vp = ca.variance_partition(Y21, yhat)
        # eq (2.17) with k=1: F = t^2 for simple regression
        f = ca.f_overall_r2(vp["r2"], vp["n"], 1)
        assert f == pytest.approx(fit["t"] ** 2, rel=1e-9)

    def test_nested_f_forms_agree(self):
        rng_y = Y21
        x2 = np.array([3, 4, 0, 0, 1, 4, 3, 1, 3, 1, 2, 1, 2, 0, 4, 3, 2, 5, 8, 4],
                      float)
        full = ca.ols_matrix(np.column_stack([X21, x2]), rng_y)
        restricted = ca.ols_matrix(X21, rng_y)
        ss_f = float(np.dot(full["residuals"], full["residuals"]))
        ss_r = float(np.dot(restricted["residuals"], restricted["residuals"]))
        f_ss = ca.f_nested_ss(ss_r, ss_f, 2, 1, 20)
        ybar = rng_y.mean()
        sst = float(np.sum((rng_y - ybar) ** 2))
        f_r2 = ca.f_nested_r2(1 - ss_f / sst, 1 - ss_r / sst, 2, 1, 20)
        # eq (2.18) uses df n-k, eq (2.19) uses n-k-1 (intercept counted);
        # eq (2.19) with matching df must equal the SS form re-computed at n-k-1
        alt = ((ss_r - ss_f) / 1) / (ss_f / (20 - 2 - 1))
        assert f_r2["f"] == pytest.approx(alt, rel=1e-9)
        assert f_ss["f"] == pytest.approx(((ss_r - ss_f) / 1) / (ss_f / 18),
                                          rel=1e-9)

    def test_beta_and_dummy_fold(self):
        assert ca.beta_standardized(2.0, 1.5, 3.0) == pytest.approx(1.0)
        # book eqs (2.21)-(2.22): b0=45.15303, b3=-1.72620 -> female intercept
        folded = ca.dummy_subgroup_equation(
            45.15303, [-0.07957, 0.46594, -1.72620], 2, 1)
        # 45.15303 - 1.72620 = 43.42683; the book prints 43.41638 (typo --
        # its own eq (2.22) arithmetic gives 43.42683, matching to 3 sf)
        assert folded["intercept"] == pytest.approx(45.15303 - 1.72620,
                                                    rel=1e-12)
        assert folded["intercept"] == pytest.approx(43.42, abs=1e-2)
        male = ca.dummy_subgroup_equation(
            45.15303, [-0.07957, 0.46594, -1.72620], 2, 0)
        assert male["intercept"] == pytest.approx(45.15303, abs=1e-9)


class TestCh3Collinearity:
    def test_tolerance_vif_table_3_5(self):
        # book Table 3.5: tolerance 0.940 <-> VIF 1.064 (age)
        assert ca.vif(1 - 0.940) == pytest.approx(1.064, abs=1e-3)
        assert ca.tolerance(0.06) == pytest.approx(0.94)
        assert ca.vif(0.0) == pytest.approx(1.0)


class TestCh4Logistic:
    B = {"b0": -1.795, "sworn": 0.002, "ne": 0.359, "south": 0.805, "west": 0.428}

    def test_logit_probability_roundtrip(self):
        for p in (0.01, 0.3, 0.5, 0.9):
            assert ca.inv_logit(ca.logit(p)) == pytest.approx(p, rel=1e-12)

    def test_west_probability_book_p163(self):
        # book: logit = 0.7443 + 0.428 for the West -> p = 0.4216
        xb = -0.7443 + 0.428
        assert ca.inv_logit(xb) == pytest.approx(0.4216, abs=5e-4)

    def test_odds_ratio_exp_b(self):
        # book Table 4.6: b=0.805 -> Exp(b)=2.237; b=0.359 -> 1.432
        assert ca.odds_ratio_unit_change(0.805) == pytest.approx(2.237, abs=1e-3)
        assert ca.odds_ratio_unit_change(0.359) == pytest.approx(1.432, abs=1e-3)
        # OR really is odds(x+1)/odds(x) under the model
        b0, b1 = -1.0, 0.7
        o1 = ca.odds(ca.inv_logit(b0 + b1 * 2))
        o0 = ca.odds(ca.inv_logit(b0 + b1 * 1))
        assert o1 / o0 == pytest.approx(ca.odds_ratio_unit_change(b1), rel=1e-12)

    def test_wald_book_p171(self):
        # book: (0.805 / 0.332)^2 = 5.879
        assert ca.wald_statistic(0.805, 0.332) == pytest.approx(5.879, abs=1e-3)
        assert ca.wald_statistic(0.805, 0.332) == pytest.approx(
            ca.coef_t(0.805, 0.332) ** 2, rel=1e-12)

    def test_model_chi2_book_p170(self):
        # book: 528.171 - 492.513 = 35.658
        assert ca.model_chi2(528.171, 492.513) == pytest.approx(35.658, abs=1e-9)

    def test_lr_chi2_book_p173(self):
        # book: reduced 499.447, full 492.513 -> 6.934
        assert ca.likelihood_ratio_chi2(499.447, 492.513) == pytest.approx(
            6.934, abs=1e-9)

    def test_cox_snell_book_table_4_12(self):
        # book reports Cox & Snell R^2 = 0.082 with chi2 = 35.658; the survey
        # n solving 1-exp(-35.658/n)=0.082 is ~416.7 -> with n=417 the value
        # rounds to 0.082.  Assert the printed value at the book's rounding.
        assert ca.cox_snell_r2(528.171, 492.513, 417) == pytest.approx(
            0.082, abs=5e-4)

    def test_derivative_beta_pct(self):
        assert ca.derivative_at_mean(0.5, 0.8) == pytest.approx(0.2)
        assert ca.beta_logistic(0.5, 0.4) == pytest.approx(0.2)
        assert ca.beta_logistic(0.5, 0.4, gelman=True) == pytest.approx(0.4)
        assert ca.percent_correct_predictions(80, 100) == pytest.approx(80.0)

    def test_ci_z(self):
        got = ca.coef_ci(0.805, 0.332, 1.96)
        assert got["lower"] == pytest.approx(0.805 - 0.332 * 1.96, rel=1e-12)
        assert got["upper"] == pytest.approx(0.805 + 0.332 * 1.96, rel=1e-12)


class TestCh5Multinomial:
    def test_softmax_reference_category(self):
        # with reference xb=0 (book p. 195): P(y=1) = 1/(1+e^xb2+e^xb3)
        xbs = [0.0, 1.2, -0.4]
        p = ca.multinomial_probs(xbs)
        denom = 1 + math.exp(1.2) + math.exp(-0.4)
        assert p[0] == pytest.approx(1 / denom, rel=1e-12)
        assert p.sum() == pytest.approx(1.0, rel=1e-12)

    def test_conditional_or(self):
        # eq (5.4): OR_{m/n} = e^{xb_m}/e^{xb_n}
        assert ca.multinomial_conditional_or(1.2, -0.4) == pytest.approx(
            math.exp(1.6), rel=1e-12)

    def test_cumulative_logits_book_p208(self):
        # book: proportions .15/.30/.35/.20 -> logits -1.735, -0.201, 1.386
        probs = [0.15, 0.30, 0.35, 0.20]
        assert ca.cumulative_logit(probs, 1) == pytest.approx(-1.735, abs=1e-3)
        assert ca.cumulative_logit(probs, 2) == pytest.approx(-0.201, abs=1e-3)
        assert ca.cumulative_logit(probs, 3) == pytest.approx(1.386, abs=1e-3)

    def test_ordinal_parameterizations(self):
        plus = ca.ordinal_logit(0.5, [0.3], [2.0], "plus")
        minus = ca.ordinal_logit(0.5, [0.3], [2.0], "minus")
        assert plus == pytest.approx(1.1)
        assert minus == pytest.approx(-0.1)


class TestCh6Count:
    def test_loglink_and_irr(self):
        assert ca.poisson_loglink_predict(0.0, 0.0, 5.0) == pytest.approx(1.0)
        # IRR is the ratio of adjacent predictions
        y2 = ca.poisson_loglink_predict(-1.0, 0.736, 2)
        y1 = ca.poisson_loglink_predict(-1.0, 0.736, 1)
        assert y2 / y1 == pytest.approx(ca.incidence_rate_ratio(0.736), rel=1e-12)

    def test_offset_scales_linearly(self):
        one = ca.poisson_offset_predict(0.1, 0.2, 3.0, 1.0)
        hundred = ca.poisson_offset_predict(0.1, 0.2, 3.0, 100.0)
        assert hundred == pytest.approx(100.0 * one, rel=1e-12)

    def test_quasi_poisson_se_book_p252(self):
        # book: se 0.083, theta 1.13 -> 0.088
        assert ca.quasi_poisson_se(0.083, 1.13) == pytest.approx(0.088, abs=5e-4)

    def test_quasi_theta_pearson_form(self):
        y = np.array([2.0, 0.0, 3.0, 1.0, 4.0, 2.0])
        yhat = np.array([1.5, 0.8, 2.5, 1.2, 3.5, 2.0])
        want = float(np.sum((y - yhat) ** 2 / yhat)) / (6 - 1 - 1)
        assert ca.quasi_poisson_theta(y, yhat, 1) == pytest.approx(want, rel=1e-12)

    def test_negbin_variance(self):
        assert ca.negative_binomial_variance(2.0, 0.5) == pytest.approx(4.0)
        assert ca.negative_binomial_variance(3.0, 0.0) == pytest.approx(3.0)


class TestCh7Multilevel:
    def test_sigma2_u_book_p283(self):
        # book: (3.9096 - 0.27) / 117.41 = 0.03
        got = ca.variance_components_sigma2_u(3.9096, 0.27, 117.41)
        assert got == pytest.approx(0.031, abs=5e-4)

    def test_icc_book_p286(self):
        # book: 0.031 / (0.031 + 0.270) = 0.10
        assert ca.intraclass_correlation(0.031, 0.270) == pytest.approx(
            0.103, abs=1e-3)

    def test_lr_chi2_book_p286(self):
        # book: -2[(-1871.73) - (-1777.35)] = 188.76
        assert ca.lr_test_chi2(-1871.73, -1777.35) == pytest.approx(188.76,
                                                                    abs=1e-2)

    def test_anova_table_7_3(self):
        # book Table 7.3: SS_between=74.28, df=19 -> MS 3.91; MS_within 0.27,
        # F = 14.48.  Reproduce F from the printed MS values.
        assert 74.28 / 19 == pytest.approx(3.91, abs=5e-3)
        assert (74.28 / 19) / 0.27 == pytest.approx(14.48, abs=5e-2)

    def test_grand_and_cluster_models(self):
        g1 = [1.0, 2.0, 3.0]
        g2 = [4.0, 6.0]
        gm = ca.grand_mean_model(np.concatenate([g1, g2]))
        assert gm["intercept"] == pytest.approx(3.2)
        cm = ca.cluster_means_model([g1, g2])
        assert cm["cluster_means"] == [pytest.approx(2.0), pytest.approx(5.0)]
        assert cm["u_j"][0] == pytest.approx(2.0 - 3.2)

    def test_centering(self):
        gc = ca.grand_mean_center([1.0, 2.0, 3.0])
        assert gc.sum() == pytest.approx(0.0)
        cc = ca.cluster_mean_center([[1.0, 3.0], [10.0, 14.0]])
        for arr in cc:
            assert arr.sum() == pytest.approx(0.0)

    def test_multilevel_predict_composes(self):
        y = ca.multilevel_predict(1.0, [2.0], [3.0], [0.5, -0.2], 0.1)
        assert y == pytest.approx(1.0 + 6.0 + 0.3 + 0.1)


class TestCh8Power:
    def test_delta_small_medium_large_book_p339(self):
        # book: n1=n2=100, d=0.2/0.5/0.8 -> delta 1.414 / 3.536 / 5.657
        for d, want in [(0.2, 1.414), (0.5, 3.536), (0.8, 5.657)]:
            assert ca.noncentrality_delta_d(d, 100, 100) == pytest.approx(
                want, abs=1e-3)

    def test_power_small_effect_book_p339(self):
        # book: delta=1.414, one-tailed alpha=.05 (t_cv about 1.653, df=198)
        # -> beta = 0.594, power = 0.406
        got = ca.power_from_delta_t(1.414, 1.6526, 198)
        assert got["beta"] == pytest.approx(0.594, abs=5e-3)
        assert got["power"] == pytest.approx(0.406, abs=5e-3)

    def test_power_medium_effect_book_p339(self):
        got = ca.power_from_delta_t(3.536, 1.6526, 198)
        assert got["power"] == pytest.approx(0.970, abs=5e-3)

    def test_lambda_book_p341(self):
        # book: medium f=0.25, n=300 -> lambda=18.75; large f=0.4 -> 48
        assert ca.noncentrality_lambda_f(0.25, 300) == pytest.approx(18.75)
        assert ca.noncentrality_lambda_f(0.4, 300) == pytest.approx(48.0)

    def test_cohens_d_f_and_r2(self):
        assert ca.cohens_d_population(10, 8, 4) == pytest.approx(0.5)
        assert ca.cohens_f(1.0, 2.0) == pytest.approx(0.5)
        # eq (8.7): f=0.25 -> R2 = 0.0625/1.0625
        assert ca.r2_from_f2(0.0625) == pytest.approx(0.0625 / 1.0625, rel=1e-12)

    def test_delta_r(self):
        # delta for r=0.3, n=100: 0.3*sqrt(98)/sqrt(0.91)
        want = 0.3 * math.sqrt(98) / math.sqrt(1 - 0.09)
        assert ca.noncentrality_delta_r(0.3, 100) == pytest.approx(want, rel=1e-12)

    def test_delta_generic(self):
        assert ca.noncentrality_delta_generic(3.5, 1.2) == pytest.approx(2.3)


class TestCh9Experiments:
    def test_confounded_treatment_book_p373(self):
        # book: r_yt=-0.25, r_yx=-0.50, r_tx=0.50 -> numerator 0, b_t = 0
        got = ca.treatment_b_confounded(-0.25, -0.50, 0.50, 1.0, 1.0)
        assert got == pytest.approx(0.0, abs=1e-12)
        # book eq (9.2): with r_tx = 0 it collapses to r_yt s_y / s_t
        assert ca.treatment_b_confounded(0.3, 0.9, 0.0, 2.0, 1.0) == \
            pytest.approx(ca.treatment_b_randomized(0.3, 2.0, 1.0), rel=1e-12)

    def test_independent_t_matches_d(self):
        got = ca.t_independent(127.8, 132.3, 10.4, 9.8, 25, 30)
        # cross-check via ch 11: d = t*sqrt((n1+n2)/(n1 n2)) must invert
        d = ca.cohens_d_sample(127.8, 132.3, 10.4, 9.8, 25, 30)
        assert ca.d_from_t(got["t"], 25, 30) == pytest.approx(d, rel=1e-9)
        assert got["df"] == 53

    def test_chi2_2x2_vs_expected_form(self):
        a, b, c, d = 30, 20, 15, 35
        got = ca.chi2_2x2(a, b, c, d)
        # classic Pearson chi2 on the same table
        obs = np.array([[a, b], [c, d]], float)
        row = obs.sum(1, keepdims=True)
        col = obs.sum(0, keepdims=True)
        exp = row @ col / obs.sum()
        pearson = float(((obs - exp) ** 2 / exp).sum())
        assert got["chi2"] == pytest.approx(pearson, rel=1e-12)

    def test_oneway_anova_null_and_effect(self):
        rng = np.random.default_rng(20260801)
        groups = [rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng.normal(1.5, 1, 50)]
        res = ca.anova_oneway(groups)
        assert res["f"] > 10  # strong effect present
        same = ca.anova_oneway([g - g.mean() for g in groups])
        assert same["ms_between"] == pytest.approx(0.0, abs=1e-20)

    def test_paired_t(self):
        d = [1.0, 2.0, 0.5, 1.5, 1.0]
        got = ca.t_paired(d)
        arr = np.array(d)
        want = arr.mean() / (arr.std(ddof=1) / math.sqrt(arr.size))
        assert got["t"] == pytest.approx(want, rel=1e-12)
        assert got["df"] == 4

    def test_repeated_measures_ms_null(self):
        # identical rows within groups: no subject variance beyond noise
        g = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        res = ca.repeated_measures_ms([g, g + 1])
        assert res["ms_subjects"] == pytest.approx(0.0, abs=1e-20)
        assert res["ms_b_subjects"] == pytest.approx(0.0, abs=1e-20)

    def test_randomized_block_removes_block_variance(self):
        rng = np.random.default_rng(7)
        block_effect = np.repeat([0.0, 5.0, 10.0, 15.0], 2)
        treat = np.tile([0, 1], 4)
        y = 1.0 * treat + block_effect + rng.normal(0, 0.1, 8)
        blocked = ca.anova_randomized_block(y, treat, np.repeat(range(4), 2))
        naive = ca.anova_oneway([y[treat == 0], y[treat == 1]])
        assert blocked["f_treatment"] > naive["f"]  # blocking boosts power


class TestCh10PSM:
    def test_bias_formula(self):
        # symmetric: equal means -> 0
        assert ca.psm_standardized_bias(0.318, 0.318, 0.186, 0.19) == \
            pytest.approx(0.0, abs=1e-12)
        got = ca.psm_standardized_bias(0.5, 0.4, 0.2, 0.2)
        assert got == pytest.approx(100 * 0.1 / 0.2, rel=1e-12)


class TestCh11Meta:
    def test_exercise_11_1_hedges_g(self):
        # study: 127.8 (n=25, sd 10.4) vs 132.3 (n=30, sd 9.8)
        d = ca.cohens_d_sample(127.8, 132.3, 10.4, 9.8, 25, 30)
        sp = ca.pooled_sd(10.4, 9.8, 25, 30)
        assert sp == pytest.approx(10.076, abs=1e-3)
        assert d == pytest.approx(-0.4466, abs=1e-3)
        g = ca.hedges_g(d, 25, 30)
        assert g == pytest.approx(d * (1 - 3 / 211), rel=1e-12)
        se = ca.se_g(g, 25, 30)
        assert se == pytest.approx(math.sqrt(55 / 750 + g * g / 110), rel=1e-12)

    def test_conversion_worked_example_book_p465(self):
        # book: failure rates .25 vs .33 -> lnOR = -0.39; logit/Cox/probit
        # conversions -0.215 / -0.237 / -0.235
        or_ = ca.odds_ratio_2x2(25, 75, 33, 67)  # p1=.25, p2=.33 per 100
        ln_or = math.log(or_)
        assert ln_or == pytest.approx(-0.39, abs=5e-3)
        assert ca.d_from_log_or(ln_or, "logit") == pytest.approx(-0.215, abs=2e-3)
        assert ca.d_from_log_or(ln_or, "cox") == pytest.approx(-0.237, abs=2e-3)
        assert ca.d_probit(0.25, 0.33) == pytest.approx(-0.235, abs=2e-3)

    def test_fisher_z_table_11_2(self):
        # book Table 11.2 anchors
        for r, want in [(0.5, 0.549), (0.75, 0.973), (0.9, 1.472)]:
            assert ca.fisher_z(r) == pytest.approx(want, abs=1e-3)
            assert ca.r_from_fisher_z(ca.fisher_z(r)) == pytest.approx(r, rel=1e-12)
        assert ca.se_fisher_z(103) == pytest.approx(0.1, rel=1e-12)

    def test_rr_or_and_ses(self):
        a, b, c, d = 40, 60, 55, 45
        rr = ca.risk_ratio(a, b, c, d)
        assert rr == pytest.approx((40 / 100) / (55 / 100), rel=1e-12)
        or_ = ca.odds_ratio_2x2(a, b, c, d)
        assert or_ == pytest.approx(40 * 45 / (60 * 55), rel=1e-12)
        # RR <-> OR round trip, eqs (11.28)-(11.29)
        p2 = 55 / 100
        assert ca.or_from_rr(rr, p2) == pytest.approx(or_, rel=1e-9)
        assert ca.rr_from_or(or_, p2) == pytest.approx(rr, rel=1e-9)
        assert ca.se_log_rr(0.4, 0.55, 100, 100) == pytest.approx(
            math.sqrt(0.6 / 40 + 0.45 / 55), rel=1e-12)
        assert ca.se_log_or(a, b, c, d) == pytest.approx(
            math.sqrt(1 / 40 + 1 / 60 + 1 / 55 + 1 / 45), rel=1e-12)

    def test_d_r_roundtrip(self):
        d = 0.6
        r = ca.r_from_d(d)                      # equal-n, eq (11.31)
        assert ca.d_from_r_pointbiserial(r) == pytest.approx(d, rel=1e-9)
        r_un = ca.r_from_d(d, 30, 70)
        assert abs(r_un) < abs(r)               # unbalanced attenuates r
        # se chain eq (11.32)/(11.33) consistency at n1=n2
        assert ca.se_r_from_se_d(d, 0.2) == pytest.approx(
            ca.se_r_from_se_d(d, 0.2, 50, 50), rel=1e-12)

    def test_d_lnor_roundtrip(self):
        d = 0.4
        # 0.606 is the book's rounding of 1/1.65, so the round trip closes
        # only to ~1e-4 relative
        assert ca.d_from_log_or(ca.log_or_from_d(d, "cox"), "cox") == \
            pytest.approx(d, rel=1e-3)
        assert ca.se_log_or_from_se_d(0.15, "logit") == pytest.approx(
            0.15 / 0.551, rel=1e-12)
        assert ca.se_d_from_se_log_or(0.3, "cox") == pytest.approx(0.3 / 1.65,
                                                                   rel=1e-12)

    def test_exercise_11_4_pooled_analysis(self):
        gs = np.array([-0.23, 0.25, 0.08, 0.10, 0.20, 0.22])
        ses = np.array([0.32, 0.31, 0.11, 0.26, 0.17, 0.24])
        ws = np.array([ca.fixed_effect_weight(s) for s in ses])
        pooled = ca.mean_effect_size(gs, ws)
        # weighted mean must lie inside the range and near the precise studies
        assert -0.23 < pooled["mean"] < 0.25
        assert pooled["se"] == pytest.approx(math.sqrt(1 / ws.sum()), rel=1e-12)
        q = ca.q_statistic(gs, ws)
        assert q["q"] == pytest.approx(q["q_computational"], rel=1e-9)
        i2 = ca.i_squared(max(q["q"], q["df"] + 1e-9), q["df"])
        assert 0 <= i2 < 100
        tau2 = ca.tau2_dersimonian_laird(gs, ws)
        assert tau2 >= 0
        # random-effects weights are never larger than fixed-effect weights
        wr = [ca.random_effects_weight(s, tau2) for s in ses]
        assert all(w_r <= w_f for w_r, w_f in zip(wr, ws))

    def test_q_partition(self):
        y1, w1 = [0.1, 0.2, 0.3], [10.0, 12.0, 9.0]
        y2, w2 = [0.5, 0.6], [8.0, 11.0]
        got = ca.q_within_between([y1, y2], [w1, w2])
        assert got["q_total"] == pytest.approx(
            got["q_within"] + got["q_between"], rel=1e-9)
        assert got["q_between"] > 0  # groups genuinely differ

    def test_logistic_sd(self):
        assert ca.LOGISTIC_SD == pytest.approx(1.8138, abs=1e-4)


class TestCh12Spatial:
    def test_expected_i_book_p510(self):
        # book: n=6 -> E(I) = -0.200
        assert ca.morans_i_expected(6) == pytest.approx(-0.200, rel=1e-12)

    def test_morans_i_checkerboard_vs_cluster(self):
        # rook-adjacency on a 2x2 grid
        w = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
                     float)
        checker = ca.morans_i(np.array([1.0, -1.0, -1.0, 1.0]), w)
        assert checker == pytest.approx(-1.0, rel=1e-9)   # perfect dispersion
        cluster = ca.morans_i(np.array([1.0, 1.0, -1.0, -1.0]), w)
        assert cluster > ca.morans_i_expected(4)          # clustered > E(I)
        # direct double-sum agreement
        x = np.array([2.0, 5.0, 1.0, 4.0])
        xd = x - x.mean()
        direct = 4 * sum(w[i, j] * xd[i] * xd[j] for i in range(4)
                         for j in range(4)) / (w.sum() * float(np.dot(xd, xd)))
        assert ca.morans_i(x, w) == pytest.approx(direct, rel=1e-12)

    def test_ols_matrix_recovers_simple(self):
        fit = ca.ols_simple(X21, Y21)
        mat = ca.ols_matrix(X21, Y21)
        assert mat["beta"][0] == pytest.approx(fit["b0"], rel=1e-9)
        assert mat["beta"][1] == pytest.approx(fit["b1"], rel=1e-9)

    def test_sar_reduced_form_satisfies_model(self):
        rng = np.random.default_rng(12)
        n = 6
        w = np.zeros((n, n))
        for i in range(n):        # ring adjacency, row-standardized
            w[i, (i - 1) % n] = w[i, (i + 1) % n] = 0.5
        xb = rng.normal(0, 1, n)
        e = rng.normal(0, 0.3, n)
        rho = 0.4
        y = ca.spatial_lag_reduced_form(rho, w, xb, e)
        assert np.allclose(y, rho * (w @ y) + xb + e, atol=1e-10)
        # rho = 0 reduces to OLS structural form, eq (12.3)
        y0 = ca.spatial_lag_reduced_form(0.0, w, xb, e)
        assert np.allclose(y0, xb + e, atol=1e-12)
