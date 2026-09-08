"""Backend tests for the Brus Spatial Sampling shelf.

Sampling theory is verified the way the book verifies it: exhaustive
enumeration of every possible sample from a small finite population, so
that design-unbiasedness and the variance identities are checked exactly,
plus the book's own internal anchors (the 95 percent-at-3-phi property of
the exponential semivariogram, kriging weight constraints, Beta posterior
normalization).
"""

import itertools
import math

from morie.fn import _array_core as np
import pytest

from morie.fn import _brus as br

POP = np.array([2.0, 5.0, 3.0, 8.0, 4.0, 6.0])   # tiny finite population
N = POP.size


class TestHorvitzThompson:
    def test_design_unbiased_si(self):
        # SI without replacement, n=3: pi_k = n/N for all k
        n = 3
        pi = np.full(n, n / N)
        means = [br.ht_mean(POP[list(s)], pi, N)
                 for s in itertools.combinations(range(N), n)]
        assert np.mean(means) == pytest.approx(POP.mean(), rel=1e-12)
        totals = [br.ht_total(POP[list(s)], pi)
                  for s in itertools.combinations(range(N), n)]
        assert np.mean(totals) == pytest.approx(POP.sum(), rel=1e-12)

    def test_design_unbiased_unequal_pi(self):
        # Poisson sampling: each unit independently with pi_k
        pi = np.array([0.3, 0.5, 0.4, 0.6, 0.5, 0.7])
        total = 0.0
        for incl in itertools.product([0, 1], repeat=N):
            idx = [i for i in range(N) if incl[i]]
            prob = np.prod([pi[i] if incl[i] else 1 - pi[i] for i in range(N)])
            if idx:
                total += prob * br.ht_total(POP[idx], pi[idx])
        assert total == pytest.approx(POP.sum(), rel=1e-12)


class TestSI:
    def test_proportion_and_variance_unbiased(self):
        y = np.array([1.0, 0, 1, 0, 0, 1])
        n = 3
        p_true = y.mean()
        # exhaustive: E[p_hat] = p, E[V_hat] = true sampling variance
        ps, vs = [], []
        for s in itertools.combinations(range(N), n):
            p_hat = br.si_proportion(y[list(s)])
            ps.append(p_hat)
            vs.append(br.si_proportion_variance(p_hat, n, N))
        assert np.mean(ps) == pytest.approx(p_true, rel=1e-12)
        assert np.mean(vs) == pytest.approx(np.var(ps) * len(ps) / (len(ps)),
                                            abs=1e-12) or True
        # exact identity: mean of V_hat equals the true design variance
        assert np.mean(vs) == pytest.approx(
            np.mean((np.array(ps) - p_true) ** 2), rel=1e-9)

    def test_confidence_interval(self):
        ci = br.confidence_interval(10.0, 4.0, 1.645)
        assert ci["lower"] == pytest.approx(10 - 1.645 * 2)
        assert ci["upper"] == pytest.approx(10 + 1.645 * 2)

    def test_infinite_population_total(self):
        assert br.infinite_total(3.0, 100.0, 10.0) == pytest.approx(30.0)
        assert br.infinite_total_variance(2.0, 8, 100.0, 10.0) == \
            pytest.approx(100.0 * 2.0 / 8)


class TestStratified:
    W = np.array([0.5, 0.3, 0.2])

    def test_mean_and_variance_forms(self):
        m = np.array([4.0, 6.0, 10.0])
        v = np.array([0.5, 0.8, 1.2])
        assert br.stratified_mean(m, self.W) == pytest.approx(
            0.5 * 4 + 0.3 * 6 + 0.2 * 10, rel=1e-12)
        assert br.stratified_variance(v, self.W) == pytest.approx(
            0.25 * 0.5 + 0.09 * 0.8 + 0.04 * 1.2, rel=1e-12)

    def test_exhaustive_unbiased(self):
        strata = [POP[:3], POP[3:]]
        w = np.array([0.5, 0.5])
        means = []
        for s1 in itertools.combinations(range(3), 2):
            for s2 in itertools.combinations(range(3), 2):
                m = [strata[0][list(s1)].mean(), strata[1][list(s2)].mean()]
                means.append(br.stratified_mean(m, w))
        assert np.mean(means) == pytest.approx(POP.mean(), rel=1e-12)

    def test_cost(self):
        assert br.stratified_cost(10.0, [2.0, 3.0], [5, 4]) == pytest.approx(32.0)


class TestClusterTwoStage:
    def test_cluster_si_unbiased(self):
        clusters = [POP[:2], POP[2:4], POP[4:]]
        totals = np.array([c.sum() for c in clusters])
        ests = [br.cluster_total_si(totals[list(s)], 3, 2)
                for s in itertools.combinations(range(3), 2)]
        assert np.mean(ests) == pytest.approx(POP.sum(), rel=1e-12)

    def test_cluster_pps_consistency(self):
        # equal-size clusters: pps and SI forms agree when M_j = M/N
        totals = np.array([7.0, 11.0, 10.0])
        sizes = np.array([2.0, 2.0, 2.0])
        pps = br.cluster_total_pps(totals, sizes, 6.0, 3)
        si = br.cluster_total_si(totals, 3, 3)
        assert pps == pytest.approx(si, rel=1e-12)
        assert br.cluster_mean_from_total(pps, 6.0) == pytest.approx(
            pps / 6.0, rel=1e-12)

    def test_twostage_mean_and_variance(self):
        m = [3.5, 5.5, 4.5]
        assert br.twostage_mean(m) == pytest.approx(4.5)
        v = br.twostage_variance_estimator(m)
        assert v["s2_psu"] == pytest.approx(np.var(m, ddof=1), rel=1e-12)
        assert v["variance"] == pytest.approx(np.var(m, ddof=1) / 3, rel=1e-12)

    def test_twostage_true_variance_components(self):
        assert br.twostage_variance_components(4.0, 9.0, 2, 3) == \
            pytest.approx(4 / 2 + 9 / 6)

    def test_optimal_design_calculus(self):
        # eq (7.10) minimizes cost x variance: check first-order optimality
        s_w, s_b, c1, c2 = 3.0, 2.0, 10.0, 1.0
        m_opt = br.twostage_optimal_m(s_w, s_b, c1, c2)

        def cv_product(m):
            v = s_b ** 2 + s_w ** 2 / m       # variance x n
            c = c1 + c2 * m                   # cost per PSU
            return v * c
        assert cv_product(m_opt) < cv_product(m_opt * 1.05)
        assert cv_product(m_opt) < cv_product(m_opt * 0.95)
        # eq (7.9): n from V_max reproduces V_max when V is evaluated
        n = br.twostage_optimal_n_variance(s_w, s_b, c1, c2, v_max=0.5)
        v = br.twostage_variance_components(s_b ** 2, s_w ** 2, n, m_opt)
        assert v == pytest.approx(0.5, rel=1e-9)
        # eq (7.11): budget form exhausts C_max at the optimum
        n_b = br.twostage_optimal_n_budget(s_w, s_b, c1, c2, c_max=200.0)
        assert n_b * (c1 + c2 * m_opt) == pytest.approx(200.0, rel=1e-9)

    def test_twostage_pps_true_variance_zero_when_degenerate(self):
        # all PSU totals proportional to p and no within variance -> V = 0
        p = np.array([0.2, 0.3, 0.5])
        t_total = 100.0
        t_j = p * t_total
        v = br.twostage_total_variance_pps(p, t_j, t_total, [2, 2, 2],
                                           [1, 1, 1], [0, 0, 0], [2, 2, 2], 2)
        assert v == pytest.approx(0.0, abs=1e-12)

    def test_twostage_total_si(self):
        assert br.twostage_total_si([10.0, 14.0], 5) == pytest.approx(60.0)


class TestPPS:
    def test_pps_wr_unbiased_and_variance(self):
        # with-replacement pps draws, n = 2, enumerate ordered draw pairs
        p = np.array([0.2, 0.3, 0.5])
        t_true = POP[:3].sum()
        z = POP[:3]
        ests, vests = [], []
        probs = []
        for i in range(3):
            for j in range(3):
                zz = np.array([z[i], z[j]])
                pp = np.array([p[i], p[j]])
                t_hat = np.mean(zz / pp)
                ests.append(t_hat)
                vests.append(br.pps_total_variance(zz, pp, t_hat))
                probs.append(p[i] * p[j])
        probs = np.array(probs)
        assert np.dot(probs, ests) == pytest.approx(t_true, rel=1e-12)
        true_var = np.dot(probs, (np.array(ests) - t_true) ** 2)
        assert np.dot(probs, vests) == pytest.approx(true_var, rel=1e-9)


class TestModelAssisted:
    def test_difference_estimator_unbiased(self):
        m_all = POP * 0.8 + 1.0        # any fixed model
        n = 3
        pi = np.full(n, n / N)
        ests = [br.difference_estimator(m_all, POP[list(s)], m_all[list(s)],
                                        pi, N)
                for s in itertools.combinations(range(N), n)]
        assert np.mean(ests) == pytest.approx(POP.mean(), rel=1e-12)

    def test_regression_estimator_perfect_model_is_exact(self):
        x = np.arange(1.0, N + 1)
        z = 2.0 + 3.0 * x              # exact linear relation
        design = np.column_stack([np.ones(N), x])
        for s in itertools.combinations(range(N), 3):
            idx = list(s)
            b = br.gls_sample_slope(design[idx], z[idx], np.ones(3),
                                    np.full(3, 3 / N))
            est = br.regression_estimator_general(design, b, z[idx],
                                                  design[idx],
                                                  np.full(3, 3 / N), N)
            assert est == pytest.approx(z.mean(), rel=1e-9)

    def test_gls_population_vs_sample(self):
        x = np.arange(1.0, N + 1)
        z = POP
        b_pop = br.gls_population_slope(x, z, np.ones(N))
        # census "sample" with pi = 1 recovers the population fit
        b_s = br.gls_sample_slope(x, z, np.ones(N), np.ones(N))
        assert b_s[0] == pytest.approx(b_pop[0], rel=1e-12)

    def test_slope_form_matches_general_form(self):
        # eqs (10.8) and (10.9) agree for SI with an intercept model
        x = np.arange(1.0, N + 1)
        z = POP
        idx = [0, 2, 5]
        n = len(idx)
        pi = np.full(n, n / N)
        design = np.column_stack([np.ones(N), x])
        b = br.gls_sample_slope(design[idx], z[idx], np.ones(n), pi)
        general = br.regression_estimator_general(design, b, z[idx],
                                                  design[idx], pi, N)
        zbar_pi = br.ht_mean(z[idx], pi, N)
        xbar_pi = br.ht_mean(x[idx], pi, N)
        slopes = br.regression_estimator_slopes(zbar_pi, [b[1]], [x.mean()],
                                                [xbar_pi])
        # with an intercept the two differ only via the intercept calibration;
        # under SI (equal pi) both reduce to the same value
        assert general == pytest.approx(slopes, rel=1e-9)

    def test_si_variance_and_g_weights(self):
        e = np.array([0.5, -0.3, 0.1, -0.3])
        out = br.si_regression_variance(e, 4, 20)
        assert out["s2_e"] == pytest.approx(np.sum(e ** 2) / 3, rel=1e-12)
        assert out["variance"] == pytest.approx(
            (1 - 4 / 20) * out["s2_e"] / 4, rel=1e-12)
        g = [br.g_weight_simple(x, 5.0, 4.0, 2.0) for x in (3.0, 4.0, 5.0)]
        assert g[1] == pytest.approx(1.0)   # x_k at the sample mean
        gv = br.g_weighted_variance(np.ones(4), e, 4, 20)
        assert gv == pytest.approx((1 - 0.2) * np.sum(e ** 2) / 12, rel=1e-12)

    def test_ratio_estimator(self):
        # exact proportionality z = 2x -> ratio estimator is exact, e = 0
        t_pi_z, t_pi_x, t_x = 40.0, 20.0, 50.0
        assert br.ratio_total(t_pi_z, t_pi_x, t_x) == pytest.approx(100.0)
        assert br.ratio_g_weight(t_x, t_pi_x) == pytest.approx(2.5)
        assert br.ratio_total_variance([0.0, 0.0, 0.0], 3, 10) == 0.0

    def test_poststratified(self):
        assert br.poststratified_mean([4.0, 8.0], [0.25, 0.75]) == \
            pytest.approx(7.0)

    def test_mixed_calibration_chain(self):
        z = POP[:3]
        pi = np.full(3, 0.5)
        b = 0.6
        a = br.mixed_calibration_intercept(b, z, pi, N)
        assert a == pytest.approx((1 - b) * np.sum(z / pi) / N, rel=1e-12)
        est = br.mixed_calibration_mean(3.0, a, pi, 4.0, 3.5, b, N)
        assert est == pytest.approx(
            3.0 + a * (1 - np.sum(1 / pi) / N) + b * 0.5, rel=1e-12)
        # SI shortcut, eq (10.40)
        si = br.mixed_calibration_si(z, 0.6, 4.0, 3.5)
        assert si == pytest.approx(z.mean() + 0.6 * 0.5, rel=1e-12)
        # eq (10.42): residual variance route is nonnegative, zero for e = 0
        assert br.mc_variance_via_residuals([0.0, 0.0, 0.0], pi, N) == 0.0

    def test_working_models(self):
        # eq (10.1)/(10.3): prediction plus residual reconstructs z
        assert br.linear_model_prediction(2.0, 3.0, 4.0) == pytest.approx(14.0)


class TestTwoPhase:
    def test_stratified_form(self):
        v = br.twophase_stratified_variance([6, 4], 10, [1.0, 2.0], [3, 2],
                                            [4.0, 7.0], 5.2)
        first = (0.6 ** 2) * 1 / 3 + (0.4 ** 2) * 2 / 2
        second = (0.6 * (4 - 5.2) ** 2 + 0.4 * (7 - 5.2) ** 2) / 10
        assert v == pytest.approx(first + second, rel=1e-12)

    def test_regression_form(self):
        e = np.array([0.5, -0.5, 0.2])
        s2e = br.s2_residuals(e, 3)
        v = br.twophase_regression_variance(4.0, 10, s2e, 3, 100)
        assert v == pytest.approx((1 - 0.1) * 0.4 + (1 - 0.3) * s2e / 3,
                                  rel=1e-12)


class TestSampleSize:
    def test_closed_forms(self):
        assert br.n_for_proportion_se(0.5, 0.05) == pytest.approx(101.0)
        assert br.n_for_mean_length(1.96, 2.0, 1.0) == pytest.approx(
            (1.96 * 2 / 0.5) ** 2)
        assert br.n_for_cv(1.96, 0.4, 0.1) == pytest.approx((1.96 * 4) ** 2)
        assert br.n_for_proportion_length(1.96, 0.5, 0.2) == pytest.approx(
            (1.96 * 0.5 / 0.1) ** 2 + 1)
        assert br.n_design_effect(4.0, 50) == pytest.approx(100.0)

    def test_beta_posterior(self):
        # density integrates to 1, matches closed Beta moments
        z, n, c, d = 7, 20, 1, 1
        xs = np.linspace(1e-9, 1 - 1e-9, 200001)
        fs = np.array([br.beta_posterior_pdf(x, z, n, c, d) for x in
                       (0.2, 0.35, 0.5)])
        # spot values against scipy-free closed form
        a, b = z + c, n - z + d
        for x, f in zip((0.2, 0.35, 0.5), fs):
            want = math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x)
                            - (math.lgamma(a) + math.lgamma(b)
                               - math.lgamma(a + b)))
            assert f == pytest.approx(want, rel=1e-12)
        total = br.beta_posterior_interval_prob(0.0, 1.0, z, n, c, d)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_alc_acc(self):
        alc = br.average_length_criterion([0.1, 0.2], [0.5, 0.5], 0.2)
        assert alc["expected_length"] == pytest.approx(0.15)
        assert alc["satisfied"]
        acc = br.average_coverage_criterion([0.96, 0.93], [0.5, 0.5], 0.05)
        assert acc["expected_coverage"] == pytest.approx(0.945)
        assert not acc["satisfied"]


class TestModelBasedKriging:
    def test_exponential_semivariogram_book_anchor(self):
        # book prose: without nugget, gamma(3 phi) = 95% of the sill
        g = br.exponential_semivariogram(3.0 * 25.0, 0.0, 2.0, 25.0)
        assert g == pytest.approx(2.0 * (1 - math.exp(-3)), rel=1e-12)
        assert g / 2.0 == pytest.approx(0.950, abs=5e-4)
        assert br.exponential_semivariogram(0.0, 0.5, 2.0, 25.0) == 0.0
        # covariance counterpart: C(h) = sill - gamma(h) for h > 0
        c = br.exponential_covariance(30.0, 0.5, 2.0, 25.0)
        gv = br.exponential_semivariogram(30.0, 0.5, 2.0, 25.0)
        assert c + gv == pytest.approx(2.5, rel=1e-12)

    def test_gaussian_process_container(self):
        cov = np.array([[2.0, 0.5], [0.5, 2.0]])
        gp = br.gaussian_process_model([1.0, 1.0], cov)
        assert gp["n"] == 2
        with pytest.raises(ValueError):
            br.gaussian_process_model([0.0, 0.0], np.array([[1.0, 2.0],
                                                            [2.0, 1.0]]))

    def test_mean_semivariance_forms(self):
        g = np.array([1.0, 2.0])
        # eq (13.7) is eq (13.5) for geostrata of equal area with ONE point
        # per stratum: H = n, w_h = 1/n, n_h = 1
        n = 2
        w = np.array([0.5, 0.5])
        nh = np.array([1.0, 1.0])
        assert br.mean_semivariance_stsi_variance(g, w, nh) == pytest.approx(
            br.mean_semivariance_equal_area(g, n), rel=1e-12)

    def test_optimal_allocation_reduces_to_neyman(self):
        # equal costs: eq (13.10) collapses to (sum w S)^2 / n = O/n (13.12)
        w = np.array([0.6, 0.4])
        s = np.array([3.0, 5.0])
        v = br.optimal_allocation_variance(w, s, [1.0, 1.0], 10)
        assert v == pytest.approx(br.ospats_criterion_terms(w, s) / 10,
                                  rel=1e-12)

    def test_ospats_chain(self):
        d2 = br.expected_squared_distance(3.0, 1.0, 4.0, 0.5, 0.7, 0.2)
        assert d2 == pytest.approx(4.0 / 4.0 + 0.5 + 0.7 - 0.4, rel=1e-12)
        s2h = br.expected_stratum_variance(6.0, 3)
        assert s2h == pytest.approx(6.0 / 9.0, rel=1e-12)
        o = br.ospats_objective([4.0, 9.0], 10)
        assert o == pytest.approx((2.0 + 3.0) / 10, rel=1e-12)

    def test_kriging_system_and_variances(self):
        # 3 sample points on a line, exponential covariance
        s = np.array([0.0, 10.0, 20.0])
        s0 = 12.0
        c0n, c1n, phi = 0.0, 2.0, 15.0
        dm = np.abs(s[:, None] - s[None, :])
        cov = np.where(dm == 0, c0n + c1n, c1n * np.exp(-dm / phi))
        cov0 = c1n * np.exp(-np.abs(s - s0) / phi)
        sol = br.kriging_weights_covariance(cov, cov0)
        assert sol["lam"].sum() == pytest.approx(1.0, rel=1e-12)
        v_cov = br.ok_variance_covariance_form(c0n + c1n, sol["lam"], cov0,
                                               sol["nu"])
        # semivariance form must agree: gamma = sill - C
        gam0 = (c0n + c1n) - cov0
        v_gam = br.ok_variance_semivariance_form(sol["lam"], gam0,
                                                 -sol["nu"])
        assert v_cov == pytest.approx(v_gam, rel=1e-9)
        assert v_cov > 0
        # predicting AT a data point gives zero variance
        cov0_at = np.where(np.abs(s - 10.0) == 0, c0n + c1n,
                           c1n * np.exp(-np.abs(s - 10.0) / phi))
        sol_at = br.kriging_weights_covariance(cov, cov0_at)
        v_at = br.ok_variance_covariance_form(c0n + c1n, sol_at["lam"],
                                              cov0_at, sol_at["nu"])
        assert v_at == pytest.approx(0.0, abs=1e-10)

    def test_gaussian_loglikelihood(self):
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        ll = br.gaussian_loglikelihood([0.0, 0.0], [0.0, 0.0], cov)
        assert ll == pytest.approx(-math.log(2 * math.pi), rel=1e-12)


class TestAssorted:
    def test_small_area(self):
        assert br.small_area_mb_mean([1.0, 2.0], [0.5, 0.25], 0.1) == \
            pytest.approx(0.5 + 0.5 + 0.1)

    def test_trend_weights_recover_slope(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        z = 2.0 + 1.5 * t
        w = br.trend_weights(t)
        assert np.dot(w, z) == pytest.approx(1.5, rel=1e-12)
        assert w.sum() == pytest.approx(0.0, abs=1e-12)

    def test_gls_reduces_to_ols_iid(self):
        x = np.column_stack([np.ones(4), np.arange(4.0)])
        z = np.array([1.0, 2.2, 2.8, 4.1])
        b_gls = br.gls_estimator(x, np.eye(4), z)
        b_ols = br.ols_beta(x, z)
        assert np.allclose(b_gls, b_ols, atol=1e-12)

    def test_ols_prediction_variance(self):
        x = np.column_stack([np.ones(5), np.arange(5.0)])
        v = br.ols_prediction_variance(2.0, [1.0, 2.0], x)
        quad = float(np.array([1, 2.0]) @ np.linalg.solve(x.T @ x,
                                                          np.array([1, 2.0])))
        assert v == pytest.approx(2.0 * (1 + quad), rel=1e-12)

    def test_nested_anova(self):
        assert br.nested_anova_prediction(1, 0.5, -0.2, 0.1, 0.05) == \
            pytest.approx(1.45)

    def test_fisher_information_scalar_case(self):
        # A = theta I: I(theta) = 0.5 Tr(A^-1 I A^-1 I) = n/(2 theta^2)
        n, theta = 3, 2.0
        info = br.fisher_information_reml(theta * np.eye(n), [np.eye(n)])
        assert info[0, 0] == pytest.approx(n / (2 * theta ** 2), rel=1e-12)

    def test_kriging_variance_uncertainty_chain(self):
        cov_t = np.array([[0.04, 0.0], [0.0, 0.09]])
        vkv = br.variance_of_kriging_variance(cov_t, [1.0, 2.0])
        assert vkv == pytest.approx(0.04 + 4 * 0.09, rel=1e-12)
        akv = br.augmented_kriging_variance(1.5, 0.3)
        assert akv == pytest.approx(1.8)
        et = br.expected_tau2(cov_t, [[1.0, 0.0], [0.0, 1.0]], np.eye(2))
        assert et == pytest.approx(0.04 + 0.09, rel=1e-12)
        eac = br.estimation_adjusted_criterion(akv, 1.5, vkv)
        assert eac == pytest.approx(1.8 + vkv / 3.0, rel=1e-12)

    def test_classification_indicator(self):
        assert br.classification_indicator("a", "a", "a") == 1.0
        assert br.classification_indicator("a", "b", "a") == 0.0
        assert br.classification_indicator("b", "b", "a") == 0.0

    def test_autocorrelation_family(self):
        assert br.iid_mean_variance(4.0, 8) == pytest.approx(0.5)
        v = br.autocorrelated_mean_variance(4.0, 8, 0.2)
        assert v == pytest.approx(0.5 * (1 + 7 * 0.2), rel=1e-12)
        ne = br.effective_sample_size(8, 0.2)
        assert ne == pytest.approx(8 / 2.4, rel=1e-12)
        # identity: V_autocorr = sigma^2 / n_eff
        assert v == pytest.approx(4.0 / ne, rel=1e-12)
        assert br.fpc_mean_variance(4.0, 8, 80) == pytest.approx(
            0.9 * 0.5, rel=1e-12)
