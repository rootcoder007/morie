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
    assert abs(rb.harrell_davis(X) - 46.6180132770) < 1e-8
    assert abs(rb.harrell_davis(X, 0.25) - 23.5087932814) < 1e-8


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
    assert abs(rb.one_step_m_estimator(X) - 50.9176160000) < 1e-7


def test_pbos_matches_wrs_pbos():
    assert abs(rb.pbos(X) - 47.7142857143) < 1e-9


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
    assert abs(r["cor"] - 0.9947292172) < 1e-9


def test_winsorized_correlation_matches_wrs_wincor():
    r = rb.winsorized_correlation(XS, YS)
    assert abs(r["cor"] - 0.9935833618) < 1e-9
    assert abs(r["p_value"] - 0.0000155727) < 1e-9


def test_winsorized_correlation_uses_n_minus_2g_minus_2_df():
    r = rb.winsorized_correlation(XS, YS, tr=0.2)
    g = rb.trim_counts(len(XS), 0.2)
    assert r["df"] == len(XS) - 2 * g - 2


def test_cliff_delta_and_interval_match_wrs_cid():
    c = rb.cliff_delta(GX, GY)
    assert abs(c["delta"] - (-0.5555555556)) < 1e-9
    assert abs(c["ci"][0] - (-0.8511568140)) < 1e-8
    assert abs(c["ci"][1] - 0.0075732462) < 1e-8


def test_cliff_delta_probabilities_sum_to_one():
    c = rb.cliff_delta(GX, GY)
    total = c["P_x_less_y"] + c["P_equal"] + c["P_x_greater_y"]
    assert abs(total - 1.0) < 1e-12
    # delta = P(X>Y) - P(X<Y)
    assert abs(c["delta"]
               - (c["P_x_greater_y"] - c["P_x_less_y"])) < 1e-12


def test_brunner_munzel_matches_wrs_bmp():
    r = rb.brunner_munzel(GX, GY)
    assert abs(r["statistic"] - 2.3959644869) < 1e-8
    assert abs(r["df"] - 14.6853762121) < 1e-7
    assert abs(r["p_value"] - 0.0303742085) < 1e-8
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
    assert abs(rb.one_step_m_estimator(X) - 50.9176160000) < 1e-7
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
