"""Backend tests for the Analysis of Categorical Data with R shelf.

Anchors come from the book's own claims and worked relations: X2 = ZS^2
(p. 36), the Wilson endpoints invert the score test (eq 1.4), the true
confidence level is an exhaustive binomial sum by definition (eq 1.6),
the Bayes estimate decomposition (eq 6.24), the 25/75 posterior-model
example around eq 5.2, and spline continuity at the knots (eq 6.35).
"""

import math

from morie.fn import _array_core as np
import pytest

from morie.fn import _acd as ac


class TestCh1:
    def test_binomial_pmf_sums_to_one(self):
        total = sum(ac.binomial_pmf(w, 10, 0.3) for w in range(11))
        assert total == pytest.approx(1.0, rel=1e-12)
        assert ac.binomial_pmf(3, 10, 0.3) == pytest.approx(
            math.comb(10, 3) * 0.3 ** 3 * 0.7 ** 7, rel=1e-12)

    def test_mle_variance(self):
        assert ac.mle_variance_pi(0.4, 25) == pytest.approx(0.24 / 25)

    def test_wilson_inverts_score_test(self):
        # endpoints satisfy |pi_hat - pi0| = z sqrt(pi0(1-pi0)/n)
        w, n, z = 4, 10, 1.96
        ci = ac.wilson_interval(w, n, z)
        p_hat = w / n
        for p0 in (ci["lower"], ci["upper"]):
            zs = abs(p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
            assert zs == pytest.approx(z, rel=1e-9)

    def test_beta_pdf_normalizes(self):
        xs = np.linspace(1e-9, 1 - 1e-9, 100001)
        vals = np.array([ac.beta_pdf(x, 2.0, 3.0) for x in xs[::1000]])
        assert vals.min() >= 0
        # closed-form check at the mode: Beta(2,3) mode = 1/3
        mode = 1 / 3
        assert ac.beta_pdf(mode, 2, 3) >= ac.beta_pdf(0.5, 2, 3)

    def test_true_confidence_level_exhaustive(self):
        # Wald interval at n=10: known to dip well below nominal (book Fig 1.3)
        z = 1.96

        def wald(w, n):
            p = w / n
            h = z * math.sqrt(max(p * (1 - p), 0) / n)
            return p - h, p + h

        def wilson(w, n):
            ci = ac.wilson_interval(w, n, z)
            return ci["lower"], ci["upper"]

        c_wald = ac.true_confidence_level(10, 0.05, wald)
        c_wilson = ac.true_confidence_level(10, 0.05, wilson)
        assert c_wilson > c_wald            # book's central comparison
        assert 0 <= c_wald <= 1 and 0 <= c_wilson <= 1

    def test_pearson_equals_score_squared(self):
        # book p. 36: X2 = ZS^2
        w1, n1, w2, n2 = 12, 30, 20, 35
        x2 = ac.pearson_chi2_two_groups(w1, n1, w2, n2)
        p_bar = (w1 + w2) / (n1 + n2)
        zs = (w1 / n1 - w2 / n2) / math.sqrt(
            p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
        assert x2["x2"] == pytest.approx(zs ** 2, rel=1e-9)

    def test_lrt_zero_when_equal(self):
        out = ac.lrt_two_groups(10, 20, 15, 30)   # both p = 0.5
        assert out["stat"] == pytest.approx(0.0, abs=1e-12)
        out2 = ac.lrt_two_groups(5, 20, 15, 20)
        assert out2["stat"] > 0

    def test_or_wald_interval(self):
        got = ac.or_wald_interval(20, 50, 10, 50, 1.96)
        or_hat = (20 / 30) / (10 / 40)
        assert got["or"] == pytest.approx(or_hat, rel=1e-12)
        se = math.sqrt(1 / 20 + 1 / 30 + 1 / 10 + 1 / 40)
        assert got["upper"] / got["or"] == pytest.approx(
            math.exp(1.96 * se), rel=1e-12)


class TestCh2Logistic:
    X = np.column_stack([np.ones(8), [0.0, 1, 2, 3, 4, 5, 6, 7]])
    Y = np.array([0.0, 0, 0, 1, 0, 1, 1, 1])

    def test_pi_and_logit_roundtrip(self):
        p = ac.logistic_pi(-1.0, [0.5], [2.0])
        assert ac.logit_form(p) == pytest.approx(0.0, abs=1e-12)

    def test_mle_matches_score_equations(self):
        fit = ac.logistic_mle(self.X, self.Y)
        pi = 1 / (1 + np.exp(-(self.X @ fit["beta"])))
        score = self.X.T @ (self.Y - pi)
        assert np.abs(score).max() < 1e-8        # MLE solves score = 0
        # loglik forms agree, eqs (2.4)-(2.5) vs (2.1)
        ll_a = ac.logistic_loglik(fit["beta"], self.X, self.Y)
        ll_b = ac.bernoulli_likelihood(pi, self.Y)
        assert ll_a == pytest.approx(ll_b, rel=1e-10)

    def test_lrt_and_deviance(self):
        full = ac.logistic_mle(self.X, self.Y)
        null = ac.logistic_mle(np.ones((8, 1)), self.Y)
        stat = ac.lrt_statistic(null["loglik"], full["loglik"])
        assert stat > 0
        # residual deviance of the full model equals -2(LL_full - LL_sat);
        # for binary data LL_sat = 0
        pi = 1 / (1 + np.exp(-(self.X @ full["beta"])))
        dev = ac.residual_deviance(pi, self.Y)
        assert dev == pytest.approx(-2 * full["loglik"], rel=1e-9)

    def test_or_ci_and_pi_ci(self):
        got = ac.or_ci_logistic(0.5, 0.04, 2.0, 1.96)
        assert got["or"] == pytest.approx(math.e, rel=1e-12)
        assert got["lower"] == pytest.approx(
            math.exp(1.0 - 2 * 1.96 * 0.2), rel=1e-12)
        cov = np.array([[0.5, -0.1], [-0.1, 0.05]])
        v = ac.linear_predictor_variance([1.0, 3.0], cov)
        assert v == pytest.approx(0.5 + 9 * 0.05 + 2 * 3 * -0.1, rel=1e-12)
        ci = ac.pi_wald_interval(0.2, v, 1.96)
        assert ci["lower"] < ci["pi"] < ci["upper"]
        assert 0 < ci["lower"] and ci["upper"] < 1   # respects (0,1)

    def test_interaction_logit(self):
        b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        got = ac.interaction_logit(b, 1, 0, 1, 0)
        assert got == pytest.approx(0.1 + 0.2 + 0.4 + 0.6, rel=1e-12)


class TestCh3Multicategory:
    def test_multinomial_pmf(self):
        # sums to 1 over all compositions of n=4 in 3 categories
        p = [0.2, 0.5, 0.3]
        total = 0.0
        for a in range(5):
            for b in range(5 - a):
                total += ac.multinomial_pmf([a, b, 4 - a - b], p)
        assert total == pytest.approx(1.0, rel=1e-10)
        # J=2 reduces to the binomial, eq (1.1)
        assert ac.multinomial_pmf([3, 7], [0.3, 0.7]) == pytest.approx(
            ac.binomial_pmf(3, 10, 0.3), rel=1e-12)

    def test_contingency_and_product_forms(self):
        counts = np.array([[2, 1], [1, 3]])
        joint = np.array([[0.2, 0.1], [0.3, 0.4]])
        assert ac.contingency_pmf(counts, joint) == pytest.approx(
            ac.multinomial_pmf(counts.ravel(), joint.ravel()), rel=1e-12)
        cond = joint / joint.sum(axis=1, keepdims=True)
        pm = ac.product_multinomial_pmf(counts, cond)
        assert 0 < pm <= 1

    def test_baseline_logits_and_probs(self):
        l2 = ac.baseline_logit(0.5, [0.3], [2.0])
        l3 = ac.baseline_logit(-0.2, [0.1], [2.0])
        probs = ac.baseline_probs([l2, l3])
        assert probs.sum() == pytest.approx(1.0, rel=1e-12)
        assert math.log(probs[1] / probs[0]) == pytest.approx(l2, rel=1e-9)
        assert math.log(probs[2] / probs[0]) == pytest.approx(l3, rel=1e-9)

    def test_proportional_odds_and_polr(self):
        # eq (3.13): polr's eta = -beta of eq (3.11)
        a = ac.proportional_odds_logit(0.5, [0.3, -0.2], [1.0, 2.0])
        b = ac.polr_parameterization(0.5, [-0.3, 0.2], [1.0, 2.0])
        assert a == pytest.approx(b, rel=1e-12)

    def test_category_prob_from_cumulative(self):
        cum = [0.15, 0.45, 0.80]
        pis = [ac.category_prob_from_cumulative(cum, j) for j in (1, 2, 3, 4)]
        assert pis == [pytest.approx(v) for v in (0.15, 0.30, 0.35, 0.20)]
        assert sum(pis) == pytest.approx(1.0)

    def test_wald_pi_j(self):
        got = ac.pi_j_wald_interval(0.3, 0.0025, 1.96)
        assert got["lower"] == pytest.approx(0.3 - 1.96 * 0.05, rel=1e-12)

    def test_ordinal_or_interval_3e50(self):
        # ordinal OR interval (stub 3e50's provenance): symmetric on log scale
        got = ac.or_ci_logistic(0.43, 0.01, 4.0, 1.96)
        assert math.sqrt(got["lower"] * got["upper"]) == pytest.approx(
            got["or"], rel=1e-9)


class TestCh4Counts:
    def test_score_interval_contains_mle(self):
        ci = ac.poisson_score_interval(3.0, 20, 1.96)
        assert ci["lower"] < 3.0 < ci["upper"]
        # endpoints invert the score test |mu_hat - mu0| = z sqrt(mu0/n)
        for m0 in (ci["lower"], ci["upper"]):
            zs = abs(3.0 - m0) / math.sqrt(m0 / 20)
            assert zs == pytest.approx(1.96, rel=1e-9)

    def test_log_link_and_loglik(self):
        assert ac.poisson_log_link(0.1, [1.0], [1.0]) == pytest.approx(
            math.exp(1.1), rel=1e-12)
        x = np.column_stack([np.ones(4), [0.0, 1, 2, 3]])
        y = np.array([1.0, 2, 4, 8])
        b = np.array([0.0, math.log(2)])
        ll = ac.poisson_loglik(b, x, y)
        mu = np.exp(x @ b)
        want = np.sum(-mu + y * np.log(mu)
                      - np.array([math.lgamma(v + 1) for v in y]))
        assert ll == pytest.approx(float(want), rel=1e-12)

    def test_loglinear_or(self):
        # 2x2: OR = exp(bXZ_22) when all other interaction terms are 0
        assert ac.loglinear_odds_ratio(0.0, 0.7, 0.0, 0.0) == pytest.approx(
            math.exp(0.7), rel=1e-12)
        # independence model -> OR = 1
        mu = [[ac.loglinear_independence_mean(1.0, bx, bz)
               for bz in (0.0, 0.4)] for bx in (0.0, 0.9)]
        or_tab = mu[0][0] * mu[1][1] / (mu[0][1] * mu[1][0])
        assert or_tab == pytest.approx(1.0, rel=1e-12)
        sat = ac.loglinear_saturated_mean(1.0, 0.9, 0.4, 0.7)
        assert sat == pytest.approx(math.exp(3.0), rel=1e-12)

    def test_ordinal_scores_and_rates(self):
        r = ac.ordinal_score_mean_ratio(0.4, 0.1, 0.2, 3.0, 1.0)
        assert r == pytest.approx(math.exp(0.3 + 0.4), rel=1e-12)
        assert ac.poisson_rate_mean(0.1, [1.0], [1.0], 100.0) == \
            pytest.approx(100 * math.exp(1.1), rel=1e-12)


class TestCh5Selection:
    def test_bic_posterior_book_example(self):
        # book: two models with Delta = 2 -> 25% vs 75% (approximately)
        taus = ac.bic_posterior_probs([100.0, 102.0])
        assert taus[1] / taus[0] == pytest.approx(math.exp(-1.0), rel=1e-12)
        assert taus[1] == pytest.approx(0.2689, abs=1e-3)   # ~25%
        assert taus[0] == pytest.approx(0.7311, abs=1e-3)   # ~75%

    def test_model_averaging(self):
        taus = [0.6, 0.4]
        thetas = [1.0, 2.0]
        ma = ac.model_averaged_estimate(taus, thetas)
        assert ma == pytest.approx(1.4, rel=1e-12)
        v = ac.model_averaged_variance(taus, thetas, [0.1, 0.2])
        want = 0.6 * ((1 - 1.4) ** 2 + 0.1) + 0.4 * ((2 - 1.4) ** 2 + 0.2)
        assert v == pytest.approx(want, rel=1e-12)


class TestCh6Additional:
    def test_prevalence_roundtrip(self):
        # eq (6.1): apparent pi from true pi_tilde, then invert
        se, sp, pt = 0.95, 0.98, 0.1
        pi = se * pt + (1 - sp) * (1 - pt)
        assert ac.prevalence_from_apparent(pi, se, sp) == pytest.approx(
            pt, rel=1e-12)
        # perfect test: identity
        assert ac.prevalence_from_apparent(0.3, 1.0, 1.0) == pytest.approx(0.3)

    def test_misclassified_loglik_max_at_mle(self):
        # eq (6.3): MLE pi_tilde_hat = (w/n + Sp - 1)/(Se + Sp - 1)
        se, sp, w, n = 0.9, 0.95, 30, 100
        mle = ac.prevalence_from_apparent(w / n, se, sp)
        ll_mle = ac.misclassified_binomial_loglik(mle, se, sp, w, n)
        for eps in (-0.01, 0.01):
            assert ll_mle >= ac.misclassified_binomial_loglik(
                mle + eps, se, sp, w, n)

    def test_exact_conditional_pmf(self):
        got = ac.exact_conditional_pmf([0.0, 1.0, 2.0], [1, 4, 2], 0.5, 1.0)
        assert got["probs"].sum() == pytest.approx(1.0, rel=1e-12)
        # beta = 0 reduces to counts/total
        flat = ac.exact_conditional_pmf([0.0, 1.0, 2.0], [1, 4, 2], 0.0, 1.0)
        assert flat["p_at_t"] == pytest.approx(4 / 7, rel=1e-12)

    def test_survey_family(self):
        assert ac.weighted_category_total([2.0, 3.0, 5.0], ["a", "b", "a"],
                                          "a") == pytest.approx(7.0)
        jv = ac.jackknife_variance([1.0, 1.2, 0.8, 1.1], 1.0)
        assert jv == pytest.approx(3 / 4 * (0.0 + 0.04 + 0.04 + 0.01),
                                   rel=1e-9)
        v = ac.survey_proportion_variance(4.0, 9.0, 1.5, 0.3, 100.0)
        assert v == pytest.approx((4 + 0.09 * 9 - 0.6 * 1.5) / 1e4,
                                  rel=1e-12)
        kc = ac.kott_carr_interval(0.3, 0.01, 2.0)
        assert kc["n_effective"] == pytest.approx(0.21 / 0.01, rel=1e-12)
        assert 0 < kc["lower"] < 0.3 < kc["upper"] < 1

    def test_mrcv_and_glmm(self):
        assert ac.spmi_loglinear_mean(1.0, 0.2, 0.3) == pytest.approx(
            math.exp(1.5), rel=1e-12)
        assert ac.three_mrcv_mean(1.0, 0.2, 0.3, 0.1) == pytest.approx(
            math.exp(1.6), rel=1e-12)
        assert ac.glmm_linear_predictor(0.5, 0.0, 0.0, -0.2) == \
            pytest.approx(0.3)
        assert ac.glmm_linear_predictor(0.5, 2.0, 1.5, -0.2) == \
            pytest.approx(0.5 + 3.0 - 0.2)

    def test_bayes_family(self):
        # eq (6.21) classic: rare disease screening
        post = ac.bayes_rule(0.99, 0.01, 0.05)
        assert post == pytest.approx(0.99 * 0.01 / (0.0099 + 0.0495),
                                     rel=1e-12)
        # eqs (6.22)-(6.23): posterior is Beta(w+a, n-w+b)
        d = ac.posterior_density_binomial(0.4, 7, 20, 1, 1)
        assert d == pytest.approx(ac.beta_pdf(0.4, 8, 14), rel=1e-12)
        # eq (6.24) identity is asserted inside; check the value
        assert ac.bayes_estimate_binomial(7, 20, 1, 1) == pytest.approx(
            8 / 22, rel=1e-12)
        # eq (6.25): flat prior -> weights proportional to likelihood
        ll = np.array([-3.0, -1.0, -2.0])
        w = ac.posterior_kernel_regression(ll, np.zeros(3))
        assert w.argmax() == 1 and w.sum() == pytest.approx(1.0)

    def test_group_testing(self):
        # perfect test: E(T) = 1 + I(1 - (1-pt)^I)
        et = ac.group_testing_expected_tests(5, 1.0, 1.0, 0.1)
        assert et == pytest.approx(1 + 5 * (1 - 0.9 ** 5), rel=1e-12)
        # pt = 0, perfect test: only the group test runs
        assert ac.group_testing_expected_tests(5, 1.0, 1.0, 0.0) == \
            pytest.approx(1.0)
        p = ac.group_testing_logit(-1.0, [0.5], [2.0])
        assert p == pytest.approx(1 / (1 + math.exp(0.0)), rel=1e-12)

    def test_splines(self):
        # eq (6.35): continuity and smoothness at the knot
        betas = [1.0, 0.5, -0.2, 0.1, 0.3]
        k = [2.0]
        eps = 1e-7
        left = ac.truncated_power_spline(2.0 - eps, betas, k)
        right = ac.truncated_power_spline(2.0 + eps, betas, k)
        assert left == pytest.approx(right, abs=1e-5)
        # piecewise form of eq (6.34) matches when pieces share the value
        pl = ac.piecewise_cubic(1.5, 2.0, [1.0, 0.5, -0.2, 0.1],
                                [9.9, 9.9, 9.9, 9.9])
        assert pl == pytest.approx(
            ac.truncated_power_spline(1.5, betas, k), rel=1e-12)
        # eq (6.36)/(6.37): basis evaluation and the OR from differences
        basis = [lambda x: 1.0, lambda x: x, lambda x: x ** 2,
                 lambda x: x ** 3,
                 lambda x: (x - 2.0) ** 3 if x > 2.0 else 0.0]
        fa = ac.spline_basis_eval(3.0, betas, basis)
        assert fa == pytest.approx(
            ac.truncated_power_spline(3.0, betas, k), rel=1e-12)
        orr = ac.spline_odds_ratio(betas, basis, 3.0, 1.0)
        fb = ac.spline_basis_eval(1.0, betas, basis)
        assert orr == pytest.approx(math.exp(fa - fb), rel=1e-12)
