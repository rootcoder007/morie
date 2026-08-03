"""Wilcox robust estimators checked against his OWN R implementation.

Every expected value below was produced by running the corresponding
function from WRS Rallfun-v45.R (github.com/nicebread/WRS) in R on the
same data, printed to 10 decimals.  The extracted R subset used to
generate them is kept at ledger/shelves/wrs_subset.R, and the run is
reproducible with ledger/tools/wrs_crosscheck.sh.

This is a cross-language known-answer test against the reference
implementation, which is a stronger check than any self-consistency
assertion.
"""
import math

import pytest

from morie.fn import _robust_core as rb

X = [12, 45, 23, 79, 19, 92, 30, 58, 132]
XS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
YS = [2, 4, 5, 9, 10, 13, 14, 17, 18, 21]
GX = [14.1, 11.2, 15.5, 9.8, 13.3, 12.2, 16.2, 10.7]
GY = [18.4, 14.9, 12.5, 17.7, 15.8, 19.1, 13.9, 16.6, 11.1]

TOL = 1e-9


def test_harrell_davis_matches_wrs_hd():
    assert abs(rb.harrell_davis(X) - 46.618013277041733) < 1e-12
    assert abs(rb.harrell_davis(X, 0.25) - 23.508793281445108) < 1e-12


def test_harrell_davis_is_bracketed_by_the_order_statistics():
    v = sorted(float(t) for t in X)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert v[0] <= rb.harrell_davis(X, q) <= v[-1]
    # monotone in q
    vals = [rb.harrell_davis(X, q) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_mom_matches_wrs_mom():
    assert abs(rb.mom_estimator(X) - 44.75) < TOL


def test_one_step_m_matches_wrs_onestep():
    assert abs(rb.one_step_m_estimator(X) - 50.917616000000002) < 1e-12


def test_pbos_matches_wrs_pbos():
    assert abs(rb.pbos(X) - 47.714285714285715) < 1e-12


def test_theil_sen_matches_wrs_tsp1reg():
    r = rb.theil_sen(XS, YS)
    assert abs(r["slope"] - 2.0) < TOL
    assert abs(r["intercept"] - 0.0) < TOL


def test_theil_sen_is_unmoved_by_an_outlier_that_wrecks_least_squares():
    bad = list(YS)
    bad[-1] = 500.0
    ts = rb.theil_sen(XS, bad)["slope"]
    # ordinary least squares slope, computed here for the comparison
    n = len(XS)
    mx = sum(XS) / n
    my = sum(bad) / n
    ols = (sum((XS[i] - mx) * (bad[i] - my) for i in range(n))
           / sum((t - mx) ** 2 for t in XS))
    assert abs(ts - 2.0) < 0.6         # barely moves
    assert ols > 6.0                   # least squares is dragged away


def test_percentage_bend_correlation_matches_wrs_pbcor():
    r = rb.percentage_bend_correlation(XS, YS)
    assert abs(r["cor"] - 0.99472921719372798) < 1e-12


def test_winsorized_correlation_matches_wrs_wincor():
    r = rb.winsorized_correlation(XS, YS)
    assert abs(r["cor"] - 0.99358336180582918) < 1e-12
    assert abs(r["p_value"] - 1.5572661047569269e-05) < 1e-17


def test_winsorized_correlation_uses_n_minus_2g_minus_2_df():
    r = rb.winsorized_correlation(XS, YS, tr=0.2)
    g = rb.trim_counts(len(XS), 0.2)
    assert r["df"] == len(XS) - 2 * g - 2


def test_cliff_delta_and_interval_match_wrs_cid():
    c = rb.cliff_delta(GX, GY)
    assert abs(c["delta"] - (-5.0 / 9.0)) < 1e-14
    assert abs(c["ci"][0] - (-0.8511568140198168)) < 1e-12
    assert abs(c["ci"][1] - 0.007573246151174589) < 1e-12


def test_cliff_delta_probabilities_sum_to_one():
    c = rb.cliff_delta(GX, GY)
    total = c["P_x_less_y"] + c["P_equal"] + c["P_x_greater_y"]
    assert abs(total - 1.0) < 1e-12
    # delta = P(X>Y) - P(X<Y)
    assert abs(c["delta"]
               - (c["P_x_greater_y"] - c["P_x_less_y"])) < 1e-12


def test_brunner_munzel_matches_wrs_bmp():
    r = rb.brunner_munzel(GX, GY)
    assert abs(r["statistic"] - 2.3959644869352816) < 1e-12
    assert abs(r["df"] - 14.68537621214997) < 1e-12
    assert abs(r["p_value"] - 0.030374208491219523) < 1e-12
    assert r["separated"] is False


def test_brunner_munzel_reports_complete_separation_instead_of_dividing_by_zero():
    # every y beats every x, so both within-group rank variances vanish
    r = rb.brunner_munzel([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert r["separated"] is True
    assert r["p_hat"] == 1.0
    assert math.isnan(r["p_value"])


def test_wilcoxon_mann_whitney_agrees_with_the_rank_sum_identity():
    r = rb.wilcoxon_mann_whitney(GX, GY)
    n1, n2 = len(GX), len(GY)
    # U and W are related by W = U + n1(n1+1)/2
    assert abs(r["statistic"] - (r["U"] + n1 * (n1 + 1) / 2)) < 1e-12
    # ranks over both groups sum to N(N+1)/2
    N = n1 + n2
    assert abs(sum(rb._rank(GX + GY)) - N * (N + 1) / 2) < 1e-9


def test_rank_averages_ties_like_r():
    assert rb._rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert rb._rank([5, 5, 5]) == [2.0, 2.0, 2.0]


def test_percentile_bootstrap_is_deterministic_and_brackets_the_estimate():
    a = rb.percentile_bootstrap_2group(GX, GY, nboot=400, seed=7)
    b = rb.percentile_bootstrap_2group(GX, GY, nboot=400, seed=7)
    assert a["ci"] == b["ci"] and a["p_value"] == b["p_value"]
    lo, hi = a["ci"]
    assert lo <= a["est_diff"] <= hi
    assert 0.0 <= a["p_value"] <= 1.0


def test_percentile_bootstrap_with_the_mean_recovers_a_clear_difference():
    a = [1.0] * 20
    b = [11.0] * 20
    r = rb.percentile_bootstrap_2group(
        a, b, est=lambda v, **k: sum(v) / len(v), nboot=200, seed=3)
    assert abs(r["est_diff"] - (-10.0)) < 1e-12
    assert r["ci"] == (-10.0, -10.0)


def test_the_two_mad_constants_are_distinct_and_named():
    # R's mad() uses 1.4826; the book writes MAD/0.6745 = 1.4825797...
    # They agree to 5 significant figures but are NOT the same number,
    # and leaving that implicit is how a silent bias gets in.
    assert rb.R_MAD_CONSTANT == 1.4826
    assert abs(rb.BOOK_MADN_CONSTANT - 1.0 / 0.6745) < 1e-15
    assert rb.R_MAD_CONSTANT != rb.BOOK_MADN_CONSTANT
    assert abs(rb.R_MAD_CONSTANT - rb.BOOK_MADN_CONSTANT) < 1e-4


def test_madn_is_exactly_mad_rescaled_with_the_book_constant():
    # one source of truth: madn must not drift from mad_rescaled
    for v in ([2, 4, 4, 4, 5, 5, 7, 9], X, GX):
        assert abs(rb.madn(v)
                   - rb.mad_rescaled(v, rb.BOOK_MADN_CONSTANT)) < 1e-13


def test_estimators_default_to_the_wrs_constant_and_accept_the_books():
    # default reproduces WRS exactly (checked above against R)
    assert abs(rb.one_step_m_estimator(X) - 50.917616000000002) < 1e-12
    # the book constant gives a different, also-correct answer; the
    # point is that the caller chooses rather than being surprised
    book = rb.one_step_m_estimator(X, constant=rb.BOOK_MADN_CONSTANT)
    assert abs(book - 50.9176160000) > 1e-6
    assert abs(book - 50.9175315) < 1e-5
    # explicitly passing the R constant is the same as the default
    assert abs(rb.one_step_m_estimator(X, constant=rb.R_MAD_CONSTANT)
               - rb.one_step_m_estimator(X)) < 1e-15
    assert abs(rb.mom_estimator(X, constant=rb.R_MAD_CONSTANT)
               - rb.mom_estimator(X)) < 1e-15


def test_mad_median_rule_keeps_the_books_constant():
    # eq. (2.14) is the book's rule, so it must use the book's MADN --
    # this is what the p.33 anchor (MADN = 0.7413) pins
    mask = [2, 2, 3, 3, 3, 4, 4, 4, 100000, 100000]
    r = rb.mad_median_rule(mask)
    assert abs(r["madn"] - rb.madn(mask)) < 1e-15
    assert abs(r["madn"] - 0.7413) < 5e-5


def test_norm_quantile_against_known_values():
    assert abs(rb._norm_quantile(0.975) - 1.959963985) < 1e-8
    assert abs(rb._norm_quantile(0.5)) < 1e-12
    assert abs(rb._norm_quantile(0.95) - 1.644853627) < 1e-8
    assert abs(rb._norm_quantile(0.025) + 1.959963985) < 1e-8


def test_degenerate_inputs_raise_rather_than_return_nonsense():
    with pytest.raises(ValueError):
        rb.theil_sen([1, 1, 1], [1, 2, 3])       # no distinct x
    with pytest.raises(ValueError):
        rb.harrell_davis([1, 2, 3], q=0.0)
    with pytest.raises(ValueError):
        rb.percentage_bend_correlation([1, 2], [1])


# --- one-sample and dependent-groups methods, WRS-anchored -----------
GY8 = [18.4, 14.9, 12.5, 17.7, 15.8, 19.1, 13.9, 16.6]


def test_trimmed_mean_se_matches_wrs_trimse():
    assert abs(rb.trimmed_mean_se(X) - 17.01437704011229) < 1e-12


def test_trimmed_mean_ci_matches_wrs_trimci():
    r = rb.trimmed_mean_ci(X)
    assert abs(r["estimate"] - (346.0 / 7.0)) < 1e-12
    assert abs(r["ci"][0] - 7.7958906092718152) < 1e-12
    assert abs(r["ci"][1] - 91.061252247871039) < 1e-12
    assert abs(r["p_value"] - 0.027153064050245534) < 1e-12
    assert r["df"] == len(X) - 2 * rb.trim_counts(len(X), 0.2) - 1


def test_trimmed_mean_ci_uses_the_tukey_mclaughlin_se():
    # se = sqrt(winvar) / ((1 - 2 tr) sqrt(n)), not the ordinary sd/sqrt(n)
    n = len(X)
    hand = (math.sqrt(rb.winsorized_variance(X, 0.2))
            / ((1 - 2 * 0.2) * math.sqrt(n)))
    assert abs(rb.trimmed_mean_se(X) - hand) < 1e-12
    assert abs(rb.trimmed_mean_ci(X)["se"] - hand) < 1e-12


def test_yuen_paired_matches_wrs_yuend():
    r = rb.yuen_paired(GX, GY8)
    assert abs(r["estimate"] - (-3.3833333333333364)) < 1e-12
    assert abs(r["se"] - 1.6194649322126944) < 1e-12
    assert abs(r["statistic"] - (-2.0891673947583707)) < 1e-12
    assert abs(r["df"] - 5.0) < 1e-12
    assert abs(r["p_value"] - 0.090996020752312834) < 1e-12


def test_yuen_paired_se_follows_the_q1_q2_q3_formula():
    # se = sqrt((q1 + q2 - 2 q3) / (h (h - 1))), q3 the (n-1)-scaled
    # Winsorized COVARIANCE -- recompute it here from the definition
    n = len(GX)
    h = n - 2 * rb.trim_counts(n, 0.2)
    q1 = (n - 1) * rb.winsorized_variance(GX, 0.2)
    q2 = (n - 1) * rb.winsorized_variance(GY8, 0.2)
    q3 = (n - 1) * rb.winsorized_correlation(GX, GY8, 0.2)["cov"]
    hand = math.sqrt((q1 + q2 - 2 * q3) / (h * (h - 1)))
    assert abs(rb.yuen_paired(GX, GY8)["se"] - hand) < 1e-12


def test_pairing_helps_only_when_the_winsorized_covariance_is_positive():
    # the -2 q3 term is the entire benefit of pairing.  For GX/GY8 the
    # Winsorized covariance is NEGATIVE, so pairing costs precision
    # rather than buying it -- assert the direction that actually holds.
    cov = rb.winsorized_correlation(GX, GY8, 0.2)["cov"]
    assert cov < 0
    assert rb.yuen_paired(GX, GY8)["se"] > rb.yuen_test(GX, GY8, 0.2)["se"]

    # and with genuinely positively related -- but not identical --
    # pairs the SE does shrink
    a = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    b = [13.4, 14.7, 17.9, 18.2, 21.6, 22.1, 25.8, 26.3]
    assert rb.winsorized_correlation(a, b, 0.2)["cov"] > 0
    assert rb.yuen_paired(a, b)["se"] < rb.yuen_test(a, b, 0.2)["se"]


def test_yuen_paired_reports_lockstep_pairs_instead_of_dividing_by_zero():
    # y = x + c makes q1 + q2 - 2 q3 exactly zero
    a = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    r = rb.yuen_paired(a, [t + 3.0 for t in a])
    assert r["degenerate"] is True
    assert r["se"] == 0.0
    assert math.isnan(r["statistic"])
    assert abs(r["estimate"] - (-3.0)) < 1e-12


def test_yuen_paired_requires_equal_lengths():
    with pytest.raises(ValueError):
        rb.yuen_paired([1, 2, 3], [1, 2])


def test_student_t_quantile_inverts_the_cdf():
    for df in (3, 7.5, 20, 200):
        for p in (0.01, 0.25, 0.5, 0.9, 0.975, 0.999):
            q = rb._student_t_quantile(p, df)
            assert abs(rb._student_t_cdf(q, df) - p) < 1e-9
    # standard table value
    assert abs(rb._student_t_quantile(0.975, 10) - 2.228138852) < 1e-6


def test_one_sample_bootstrap_is_deterministic_and_covers_the_estimate():
    a = rb.one_sample_bootstrap(X, nboot=500, seed=11)
    b = rb.one_sample_bootstrap(X, nboot=500, seed=11)
    assert a["ci"] == b["ci"]
    lo, hi = a["ci"]
    assert lo <= a["estimate"] <= hi
    assert 0.0 <= a["p_value"] <= 1.0


def test_one_sample_bootstrap_rejects_a_far_null():
    r = rb.one_sample_bootstrap(X, nboot=500, seed=5, null_value=1e6)
    assert r["p_value"] == 0.0


# --- one-way ANOVA, outlier rules, effect size (WRS-anchored) -------
G1 = [14.1, 11.2, 15.5, 9.8, 13.3, 12.2, 16.2, 10.7]
G2 = [18.4, 14.9, 12.5, 17.7, 15.8, 19.1, 13.9, 16.6, 11.1]
G3 = [21.2, 19.8, 24.1, 17.3, 22.5, 20.4, 18.9, 23.7]


def test_trimmed_mean_anova_matches_wrs_t1way():
    r = rb.trimmed_mean_anova([G1, G2, G3])
    assert abs(r["statistic"] - 17.861507627355412) < 1e-12
    assert abs(r["df1"] - 2.0) < 1e-12
    assert abs(r["df2"] - 10.628521853986074) < 1e-12
    assert abs(r["p_value"] - 0.00039906634158703991) < 1e-16


def test_trimmed_mean_anova_on_two_groups_agrees_with_yuen():
    # with J = 2 the Welch-type F is the square of Yuen's t
    f = rb.trimmed_mean_anova([G1, G2])
    t = rb.yuen_test(G1, G2, tr=0.2)
    assert abs(f["statistic"] - t["statistic"] ** 2) < 1e-8
    assert abs(f["p_value"] - t["p_value"]) < 1e-9


def test_trimmed_mean_anova_rejects_medians_and_tiny_groups():
    with pytest.raises(ValueError):
        rb.trimmed_mean_anova([G1, G2], tr=0.5)
    with pytest.raises(ValueError):
        rb.trimmed_mean_anova([G1])
    with pytest.raises(ValueError):      # zero Winsorized variance
        rb.trimmed_mean_anova([[1, 1, 1, 1, 1, 1], G1])


def test_boxplot_rule_matches_wrs_outbox():
    r = rb.boxplot_outliers(X)
    assert abs(r["lower"] - (-70.833333333333314)) < 1e-12
    assert abs(r["upper"] - 175.83333333333331) < 1e-12


def test_carling_rule_matches_wrs_outbox_mbox():
    r = rb.boxplot_outliers(X, carling=True)
    assert abs(r["lower"] - (-81.260045489006785)) < 1e-12
    assert abs(r["upper"] - 171.26004548900679) < 1e-12
    # Carling's fence depends on n; the plain rule's does not
    n = len(X)
    assert abs(r["gval"] - (17.63 * n - 23.64) / (7.74 * n - 3.71)) < 1e-12
    assert rb.boxplot_outliers(X)["gval"] == 1.5


def test_carling_centres_on_the_median_not_the_quartiles():
    r = rb.boxplot_outliers(X, carling=True)
    m = rb.median(X)
    assert abs((r["lower"] + r["upper"]) / 2 - m) < 1e-9
    # the plain rule is centred on the midpoint of the fourths instead
    b = rb.boxplot_outliers(X)
    f = rb.ideal_fourths(X)
    assert abs((b["lower"] + b["upper"]) / 2
               - (f["q1"] + f["q2"]) / 2) < 1e-9


def test_akp_effect_size_matches_r_integrate_not_mass_area():
    # WRS akp.effect scales by area(dnormvar, ...), i.e. MASS::area, an
    # adaptive Simpson whose tolerance leaves it 9.9e-10 short of the
    # true integral.  Our adaptive Gauss-Kronrod agrees with R's
    # integrate() to 3.6e-16, so we match the ACCURATE value:
    #   MASS::area  -> -0.836340302559   (what WRS prints)
    #   integrate   -> -0.836340301553   (correct, and ours)
    r = rb.akp_effect_size(G1, G2)
    assert abs(r["effect_size"] - (-0.836340301553)) < 1e-11
    assert abs(r["effect_size"] - (-0.836340302559)) > 1e-10


def test_adaptive_quadrature_matches_r_integrate_to_machine_precision():
    lo = rb._norm_quantile(0.2)
    hi = rb._norm_quantile(0.8)

    def f(u):
        return u * u * math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)

    # R: integrate(function(u) u^2*dnorm(u), qnorm(.2), qnorm(.8))
    assert abs(rb._adaptive_quad(f, lo, hi)
               - 0.12875620638587673) < 1e-15
    # a polynomial the rule must integrate exactly
    assert abs(rb._adaptive_quad(lambda u: u ** 3, 0.0, 2.0) - 4.0) < 1e-12


def test_akp_cterm_is_one_without_trimming_and_recovers_cohens_d():
    # with tr = 0 the rescaling constant is 1 and the estimator reduces
    # to the ordinary pooled-variance Cohen's d
    r = rb.akp_effect_size(G1, G2, tr=0.0)
    assert abs(r["cterm"] - 1.0) < 1e-12
    n1, n2 = len(G1), len(G2)
    m1 = sum(G1) / n1
    m2 = sum(G2) / n2
    sp = math.sqrt(((n1 - 1) * rb.variance(G1)
                    + (n2 - 1) * rb.variance(G2)) / (n1 + n2 - 2))
    assert abs(r["effect_size"] - (m1 - m2) / sp) < 1e-12


def test_akp_unequal_variance_returns_one_value_per_group():
    r = rb.akp_effect_size(G1, G2, equal_variance=False)
    assert isinstance(r["effect_size"], tuple)
    assert len(r["effect_size"]) == 2
    assert all(v < 0 for v in r["effect_size"])   # G1 sits below G2


def test_f_cdf_against_known_values():
    # F(1; d, d) = 0.5 by symmetry of the numerator and denominator
    for df in (2, 5, 30):
        assert abs(rb._f_cdf(1.0, df, df) - 0.5) < 1e-9
    assert rb._f_cdf(0.0, 3, 7) == 0.0
    # an F with df1 = 1 is a squared t
    assert abs(rb._f_cdf(4.0, 1, 12)
               - (2 * rb._student_t_cdf(2.0, 12) - 1)) < 1e-9


def test_trimmed_mean_bootstrap_covers_the_trimmed_mean():
    r = rb.trimmed_mean_bootstrap(X, nboot=500, seed=4)
    lo, hi = r["ci"]
    assert lo <= rb.trimmed_mean(X, 0.2) <= hi
    assert abs(r["estimate"] - rb.trimmed_mean(X, 0.2)) < 1e-12


# --- median standard error and Winsorized regression (WRS-anchored) --
def test_median_se_matches_wrs_msmedse():
    assert abs(rb.median_se(X, warn_ties=False)["se"]
               - 23.293468987767866) < 1e-12
    assert abs(rb.median_se(G1, warn_ties=False)["se"]
               - 1.2423183460142859) < 1e-12


def test_median_se_follows_the_mckean_shrader_order_statistics():
    v = sorted(float(t) for t in X)
    n = len(v)
    z = rb._norm_quantile(0.995)
    av = int(round((n + 1) / 2 - z * math.sqrt(n / 4)))
    top = n - av + 1
    r = rb.median_se(X, warn_ties=False)
    assert r["av"] == av and r["top"] == top
    assert abs(r["se"] - (v[top - 1] - v[av - 1]) / (2 * z)) < 1e-12


def test_median_se_reports_ties_rather_than_hiding_them():
    # Wilcox warns this estimator can be badly wrong with ties, so the
    # tie must be surfaced, not silently folded into a number
    tied = [1, 2, 2, 2, 3, 3, 4, 5, 5]
    r = rb.median_se(tied)
    assert r["ties"] is True
    assert "warning" in r
    assert rb.median_se(X, warn_ties=False)["ties"] is False


def test_median_test_2group_is_symmetric_under_swapping():
    a = rb.median_test_2group(G1, G2)
    b = rb.median_test_2group(G2, G1)
    assert abs(a["estimate"] + b["estimate"]) < 1e-12
    assert abs(a["statistic"] + b["statistic"]) < 1e-12
    assert abs(a["p_value"] - b["p_value"]) < 1e-12


def test_winsorized_regression_matches_wrs_winreg():
    w = rb.winsorized_regression(XS, YS)
    assert abs(w["intercept"] - (-0.27202587302805831)) < 1e-12
    assert abs(w["slope"][0] - 2.106668770164883) < 1e-12
    assert w["converged"] is True


def test_winsorized_regression_resists_a_y_outlier():
    bad = list(YS)
    bad[-1] = 500.0
    wr = rb.winsorized_regression(XS, bad)["slope"][0]
    n = len(XS)
    mx = sum(XS) / n
    my = sum(bad) / n
    ols = (sum((XS[i] - mx) * (bad[i] - my) for i in range(n))
           / sum((t - mx) ** 2 for t in XS))
    assert abs(wr - 2.0) < 1.0        # stays near the true slope
    assert ols > 6.0                  # least squares is dragged away


def test_winsorized_regression_accepts_multiple_predictors():
    rows = [[float(i), float(i * i)] for i in range(1, 13)]
    y = [3.0 + 2.0 * r[0] - 0.5 * r[1] for r in rows]
    w = rb.winsorized_regression(rows, y)
    assert len(w["slope"]) == 2
    assert len(w["coef"]) == 3
    fitted = [w["intercept"] + sum(w["slope"][j] * rows[i][j]
                                   for j in range(2))
              for i in range(len(rows))]
    assert max(abs(fitted[i] - y[i]) for i in range(len(y))) < 1e-6


def test_winsorized_regression_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rb.winsorized_regression([1, 2, 3], [1, 2])


# --- bootstrap correlation interval ---------------------------------
def test_correlation_bootstrap_matches_wrs_corb_exactly():
    # R: source(wrs_subset6.R); corb(xs, ys, nboot = 599)
    # Now that the resampling uses R's Mersenne-Twister stream and
    # R_unif_index, the INTERVAL matches too, not just the estimate.
    r = rb.correlation_bootstrap_ci(XS, YS, nboot=599)
    assert abs(r["estimate"] - 0.99472921719372798) < 1e-12
    assert abs(r["ci"][0] - 0.97463184619707632) < 1e-12
    assert abs(r["ci"][1] - 0.99881188076857419) < 1e-12
    assert r["p_value"] == 0.0


def test_correlation_bootstrap_wraps_the_anchored_estimator():
    w = rb.correlation_bootstrap_ci(XS, YS,
                                    corfun=rb.winsorized_correlation,
                                    nboot=299)
    assert abs(w["estimate"] - 0.99358336180582918) < 1e-12


def test_correlation_bootstrap_is_deterministic_and_brackets_the_estimate():
    a = rb.correlation_bootstrap_ci(XS, YS, nboot=299, seed=13)
    b = rb.correlation_bootstrap_ci(XS, YS, nboot=299, seed=13)
    assert a["ci"] == b["ci"] and a["p_value"] == b["p_value"]
    lo, hi = a["ci"]
    assert lo <= a["estimate"] <= hi
    assert 0.0 <= a["p_value"] <= 1.0


def test_correlation_bootstrap_resamples_pairs_not_columns():
    # a strong relationship must survive resampling; if the two vectors
    # were resampled independently the interval would straddle zero
    r = rb.correlation_bootstrap_ci(XS, YS, nboot=299, seed=1)
    assert r["ci"][0] > 0.5
    assert r["p_value"] < 0.05


def test_correlation_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rb.correlation_bootstrap_ci([1, 2, 3], [1, 2])


# --- R-compatible RNG ------------------------------------------------
def test_rrng_reproduces_r_runif():
    from morie.fn import _rrng_core as rr
    # R: set.seed(2); runif(5)
    got = rr.RRandom(2).runif(5)
    want = [0.1848822599, 0.7023740360, 0.5733263348,
            0.1680519204, 0.9438393388]
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-10
    # R: set.seed(42); runif(3)
    got = rr.RRandom(42).runif(3)
    want = [0.9148060435, 0.9370754133, 0.2861395348]
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-10


def test_rrng_reproduces_r_sample():
    from morie.fn import _rrng_core as rr
    # R: set.seed(2); sample.int(10, 10, replace = TRUE)
    assert rr.RRandom(2).sample_int(10, 10, replace=True) == \
        [5, 6, 6, 8, 1, 1, 9, 2, 1, 3]
    # R: set.seed(1); sample.int(10, 5)
    assert rr.RRandom(1).sample_int(10, 5) == [9, 4, 7, 1, 2]


def test_rrng_sample_without_replacement_is_a_permutation():
    from morie.fn import _rrng_core as rr
    out = rr.RRandom(7).sample_int(20, 20)
    assert sorted(out) == list(range(1, 21))
    with pytest.raises(ValueError):
        rr.RRandom(1).sample_int(5, 6)


def test_rrng_stream_is_reproducible_and_advances():
    from morie.fn import _rrng_core as rr
    a = rr.RRandom(99).runif(4)
    b = rr.RRandom(99).runif(4)
    assert a == b
    g = rr.RRandom(99)
    first, second = g.runif(2), g.runif(2)
    assert first != second          # the stream advances
    assert all(0.0 < v < 1.0 for v in a + first + second)
