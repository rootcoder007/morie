"""The native Kolmogorov-Smirnov family.

Reference values are R's ``ks.test`` on the same data, to 9 decimals.
R uses the exact distributions -- Marsaglia, Tsang and Wang (2003) for
the one-sample two-sided law and the Smirnov path recursion for the
two-sample one -- and so does this, which is why the agreement is exact
rather than approximate.
"""

import math

import pytest

from morie.fn import _stats_core as sc
from morie.fn import kstest


def lcg(n, seed):
    out, s = [], seed
    for _ in range(n):
        s = (1103515245 * s + 12345) % 2147483648
        out.append(s / 2147483648.0)
    return out


X10 = [0.1, 0.25, 0.3, 0.44, 0.5, 0.61, 0.7, 0.85, 0.9, 0.99]


def test_one_sample_matches_r_two_sided():
    r = sc.kstest(X10, "uniform")
    assert r.statistic == pytest.approx(0.15, abs=1e-9)
    assert r.pvalue == pytest.approx(0.953965, abs=1e-6)
    assert r.exact is True


def test_one_sample_matches_r_one_sided():
    g = sc.kstest(X10, "uniform", alternative="greater")
    assert g.statistic == pytest.approx(0.01, abs=1e-9)
    assert g.pvalue == pytest.approx(0.989063, abs=1e-6)
    ls = sc.kstest(X10, "uniform", alternative="less")
    assert ls.statistic == pytest.approx(0.15, abs=1e-9)
    assert ls.pvalue == pytest.approx(0.583128, abs=1e-6)
    assert g.exact is True and ls.exact is True


def test_the_two_sided_statistic_is_the_larger_one_sided_one():
    r = sc.kstest(X10, "uniform")
    assert r.statistic == max(r.d_plus, r.d_minus)
    assert r.d_plus >= 0 and r.d_minus >= 0


def test_two_sample_matches_r():
    a = [1, 2, 3, 4, 5, 6]
    b = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    two = sc.ks_2samp(a, b)
    assert two.statistic == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert two.pvalue == pytest.approx(0.930736, abs=1e-6)
    assert sc.ks_2samp(a, b, alternative="greater").pvalue == pytest.approx(
        0.535714, abs=1e-6)
    assert sc.ks_2samp(a, b, alternative="less").pvalue == pytest.approx(
        1.0, abs=1e-9)
    assert two.exact is True and two.n_ties == 0


def test_the_exact_law_is_used_below_n_100_and_the_series_above():
    assert sc.kstest(lcg(90, 3), "uniform").exact is True
    assert sc.kstest(lcg(120, 3), "uniform").exact is False


def test_ties_force_the_asymptotic_two_sample_p_value():
    a = [1, 2, 3, 4, 5]
    b = [3, 4, 5, 6, 7]
    r = sc.ks_2samp(a, b)
    assert r.n_ties == 3
    assert r.exact is False


def test_the_exact_kolmogorov_law_is_a_distribution():
    # nondecreasing in d, and spanning 0 to 1
    vals = [sc._ks_pkolmogorov(d / 20.0, 12) for d in range(21)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert sc._ks_pkolmogorov(0.0, 12) == 0.0
    assert sc._ks_pkolmogorov(1.0, 12) == 1.0


def test_the_smirnov_recursion_is_a_distribution():
    vals = [sc._ks_psmirnov(d / 10.0, 8, 9) for d in range(11)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    # the recursion returns P(D < d) at the largest attainable lattice
    # value below d, so at d = 1 what is left over is exactly the two
    # arrangements that separate the samples completely
    n_arrangements = math.comb(8 + 9, 8)
    assert 1.0 - vals[-1] == pytest.approx(2.0 / n_arrangements, rel=1e-9)


def test_a_perfect_fit_gives_a_small_statistic():
    n = 40
    x = [(i + 0.5) / n for i in range(n)]
    r = sc.kstest(x, "uniform")
    assert r.statistic <= 1.0 / (2 * n) + 1e-12
    assert r.pvalue > 0.99


def test_a_gross_misfit_is_rejected():
    r = sc.kstest([0.9 + 0.01 * i for i in range(10)], "uniform")
    assert r.statistic > 0.5
    assert r.pvalue < 0.01


def test_a_callable_cdf_is_accepted():
    x = lcg(30, 11)
    a = sc.kstest(x, "uniform")
    b = sc.kstest(x, lambda u: min(1.0, max(0.0, u)))
    assert a.statistic == pytest.approx(b.statistic, abs=1e-12)
    assert a.pvalue == pytest.approx(b.pvalue, abs=1e-12)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        sc.kstest(X10, "uniform", alternative="bogus")
    with pytest.raises(ValueError):
        sc.kstest(X10, "cauchy")
    with pytest.raises(ValueError):
        sc.ks_2samp([], [1.0])
    with pytest.raises(ValueError):
        sc.ks_1samp([], lambda u: u)


def test_the_public_wrapper_carries_the_alternative_through():
    # this used to raise TypeError: kstest() got an unexpected keyword
    # argument 'alternative'
    for alt in ("two-sided", "greater", "less"):
        r = kstest(X10, distribution="uniform", alternative=alt)
        assert 0.0 <= r.p_value <= 1.0
        assert r.statistic == pytest.approx(
            sc.kstest(X10, "uniform", alternative=alt).statistic, abs=1e-12)
    two = kstest(X10, [0.2, 0.4, 0.6, 0.8])
    assert two.method == "two-sample"


def test_the_normal_route_agrees_with_the_standard_normal_cdf():
    x = [-1.5, -0.4, 0.0, 0.3, 1.1, 2.2]
    r = sc.kstest(x, "norm")
    manual = sc.ks_1samp(
        x, lambda u: 0.5 * (1.0 + math.erf(u / math.sqrt(2.0))))
    assert r.statistic == pytest.approx(manual.statistic, abs=1e-12)
