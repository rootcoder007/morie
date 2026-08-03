"""Wilcox (2017) robust methods, checked against the book's own printed
worked examples.

Every expected value below is a number the book prints, with the page it
appears on.  These are known-answer tests, not self-consistency checks:
if the implementation drifts from the source, they fail.
"""
import math

import pytest

from morie.fn import _robust_core as rb


# p.27, sec 2.4.3 -- the worked ideal-fourths example
IDEALF_X = [-29.6, -20.9, -19.7, -15.4, -12.3, -8.0,
            -4.3, 0.8, 2.0, 6.2, 11.2, 25.0]

# p.28, sec 2.4.6 -- winvar(x) = 937.9, winvar(x, 0) = s^2 = 1596.8
WINVAR_X = [12, 45, 23, 79, 19, 92, 30, 58, 132]

# p.33, sec 2.5.2 -- the masking example
MASK_X = [2, 2, 3, 3, 3, 4, 4, 4, 100000, 100000]


def test_ideal_fourths_reproduce_the_books_worked_example():
    f = rb.ideal_fourths(IDEALF_X)
    # "n/4 + 5/12 = 3.41667 ... gives j = 3, so h = 0.41667"
    assert f["j"] == 3
    assert abs(f["h"] - 0.41667) < 1e-5
    # q1 = (1 - 0.41667)(-19.7) + 0.41667(-15.4).  The book prints
    # -17.9, rounded to one decimal; the exact value is -17.9092.
    assert abs(f["q1"] - (-17.9)) < 0.01
    assert abs(f["q1"] - (0.58333 * -19.7 + 0.41667 * -15.4)) < 1e-4
    # q2 = (1 - 0.41667)(6.2) + 0.41667(2) = 4.45 exactly
    assert abs(f["q2"] - 4.45) < 5e-4
    # IQR = q2 - q1; the book's 22.35 carries q1's rounding
    assert abs(rb.idealf_iqr(IDEALF_X) - 22.35) < 0.01
    assert abs(rb.idealf_iqr(IDEALF_X) - (f["q2"] - f["q1"])) < 1e-12


def test_ideal_fourths_follow_equations_2_6_and_2_7_exactly():
    # recompute (2.6)-(2.7) independently from the definition
    v = sorted(IDEALF_X)
    n = len(v)
    j = int(math.floor(n / 4 + 5 / 12))
    h = n / 4 + 5 / 12 - j
    k = n - j + 1
    f = rb.ideal_fourths(v)
    assert abs(f["q1"] - ((1 - h) * v[j - 1] + h * v[j])) < 1e-12
    assert abs(f["q2"] - ((1 - h) * v[k - 1] + h * v[k - 2])) < 1e-12


def test_winsorized_variance_matches_the_printed_value():
    # "winvar(x) returns the value 937.9, which is the 20% Winsorized
    # variance"
    assert abs(rb.winsorized_variance(WINVAR_X, 0.2) - 937.9) < 0.05


def test_no_winsorizing_gives_the_ordinary_sample_variance():
    # "winvar(x,0) returns the sample variance, s^2, which is 1596.8"
    assert abs(rb.winsorized_variance(WINVAR_X, 0.0) - 1596.8) < 0.05
    assert abs(rb.winsorized_variance(WINVAR_X, 0.0)
               - rb.variance(WINVAR_X)) < 1e-12


def test_winsorizing_shrinks_the_variance():
    # "Typically the Winsorized variance will be smaller than the sample
    # variance s^2 because Winsorizing pulls in extreme values."
    assert rb.winsorized_variance(WINVAR_X, 0.2) < rb.variance(WINVAR_X)


def test_winsorize_replaces_rather_than_discards():
    w = rb.winsorize(WINVAR_X, 0.2)
    # same sample size as the input -- trimming would have shortened it
    assert len(w) == len(WINVAR_X)
    g = rb.trim_counts(len(WINVAR_X), 0.2)
    assert g == 1                       # floor(0.2 * 9) = 1
    v = sorted(float(t) for t in WINVAR_X)
    assert w[0] == v[1] and w[-1] == v[-2]


def test_madn_and_the_mad_median_rule_match_the_masking_example():
    # "M = 3.5, MADN = MAD/0.6745 = 0.7413"
    assert abs(rb.median(MASK_X) - 3.5) < 1e-12
    assert abs(rb.madn(MASK_X) - 0.7413) < 5e-5
    # "(100000 - 3.5)/0.7413 = 134893.4".  That printed figure divides
    # by the ROUNDED MADN; carrying full precision gives 134895.28, so
    # check the book's arithmetic on its own rounded input and our own
    # against the exact MADN.
    r = rb.mad_median_rule(MASK_X)
    assert abs((100000 - 3.5) / 0.7413 - 134893.4) < 0.1
    assert abs(max(r["ratio"]) - (100000 - 3.5) / rb.madn(MASK_X)) < 1e-6
    assert abs(max(r["ratio"]) - 134893.4) < 3.0
    # "100,000 would now be declared an outlier"
    assert r["n_outliers"] == 2
    assert set(r["outliers"]) == {100000.0}


def test_mean_variance_rule_masks_where_mad_median_does_not():
    # the book's whole point: the two 100000s are NOT flagged by a
    # mean-and-variance rule, because they inflate the very variance
    # the rule divides by
    v = [float(t) for t in MASK_X]
    m = sum(v) / len(v)
    s = math.sqrt(rb.variance(v))
    assert all(abs(t - m) / s <= 2.24 for t in v)   # nothing flagged
    assert rb.mad_median_rule(v)["n_outliers"] == 2  # MAD-median flags them


def test_trimmed_mean_endpoints():
    x = [1, 2, 3, 4, 100]
    assert abs(rb.trimmed_mean(x, 0.0) - sum(x) / 5) < 1e-12
    # floor(0.2*5) = 1 trimmed per tail, leaving 2,3,4
    assert abs(rb.trimmed_mean(x, 0.2) - 3.0) < 1e-12


def test_trimmed_mean_resists_an_outlier_the_mean_does_not():
    base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    wild = base[:-1] + [10000]
    assert abs(rb.trimmed_mean(base, 0.2) - rb.trimmed_mean(wild, 0.2)) < 1e-9
    assert abs(sum(base) / 10 - sum(wild) / 10) > 900


def test_yuen_reduces_to_welch_when_there_is_no_trimming():
    # the book's defining property of Yuen's method, sec. 7.4.1:
    # "reduces to Welch's method for means when there is no trimming"
    x = [11.1, 12.2, 15.5, 9.8, 14.1, 13.3, 10.7, 16.2]
    y = [18.4, 14.9, 17.7, 12.5, 19.1, 15.8, 13.9, 20.3, 16.6]
    yu = rb.yuen_test(x, y, tr=0.0)
    we = rb.welch_test(x, y)
    assert abs(yu["statistic"] - we["statistic"]) < 1e-10
    assert abs(yu["df"] - we["df"]) < 1e-10
    assert abs(yu["p_value"] - we["p_value"]) < 1e-12


def test_yuen_is_less_disturbed_by_an_outlier_than_welch():
    x = [11.1, 12.2, 15.5, 9.8, 14.1, 13.3, 10.7, 16.2]
    y = [18.4, 14.9, 17.7, 12.5, 19.1, 15.8, 13.9, 20.3, 16.6]
    y_bad = y[:-1] + [500.0]
    d_yuen = abs(rb.yuen_test(x, y, 0.2)["statistic"]
                 - rb.yuen_test(x, y_bad, 0.2)["statistic"])
    d_welch = abs(rb.welch_test(x, y)["statistic"]
                  - rb.welch_test(x, y_bad)["statistic"])
    assert d_yuen < d_welch


def test_student_t_cdf_against_known_quantiles():
    # two-sided 5% critical values, standard tables
    assert abs(rb._student_t_cdf(2.228, 10) - 0.975) < 5e-4
    assert abs(rb._student_t_cdf(1.960, 10 ** 6) - 0.975) < 1e-3
    assert abs(rb._student_t_cdf(0.0, 7) - 0.5) < 1e-12
    # symmetry
    assert abs(rb._student_t_cdf(-1.3, 9)
               + rb._student_t_cdf(1.3, 9) - 1.0) < 1e-12


def test_boxplot_rule_flags_the_far_tail():
    x = IDEALF_X + [500.0]
    r = rb.boxplot_rule(x)
    assert 500.0 in r["outliers"]
    assert all(t not in r["outliers"] for t in IDEALF_X)


def test_trimming_bounds_are_enforced():
    with pytest.raises(ValueError):
        rb.trimmed_mean([1, 2, 3], 0.5)
    with pytest.raises(ValueError):
        rb.trimmed_mean([1, 2, 3], -0.1)
    with pytest.raises(ValueError):
        rb.ideal_fourths([1, 2])
