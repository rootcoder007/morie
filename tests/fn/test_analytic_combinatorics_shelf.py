"""Analytic combinatorics: every asymptotic held against exact values.

An asymptotic estimate is a theorem about exact coefficients, so the
tests compute the exact sequence by recurrence and check the estimate
against it -- ratio converging, error bound holding, rounding identity
exact -- rather than comparing the formula with itself.

Sources: Flajolet and Sedgewick (2009) *Analytic Combinatorics*;
Hardy and Ramanujan (1918); de Bruijn (1981).
"""

import math
from fractions import Fraction

import pytest

from morie.fn.anlcmb import (
    derangement_rounding,
    dominant_singularity_growth,
    hardy_ramanujan_partitions,
    rational_gf_coefficients,
    singularity_transfer,
    stirling_series_error,
)
from morie.fn.enumcb import catalan_number, derangements, partition_count


# --------------------------------------------------------------------
# Rational generating functions
# --------------------------------------------------------------------

def test_fibonacci_comes_out_of_its_generating_function():
    got = rational_gf_coefficients([0, 1], [1, -1, -1], 30)
    fib = [0, 1]
    while len(fib) < 30:
        fib.append(fib[-1] + fib[-2])
    assert got["coefficients"] == fib
    assert got["all_integral"] is True


def test_geometric_and_binary_string_series():
    assert rational_gf_coefficients([1], [1, -2], 10)["coefficients"] == \
        [2 ** k for k in range(10)]
    # strings over {a,b} with no two consecutive a's: 1/(1-x-x^2) shifted
    got = rational_gf_coefficients([1], [1, -1, -1], 10)["coefficients"]
    assert got == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def test_non_integer_coefficients_are_exact_fractions():
    got = rational_gf_coefficients([1], [2, -1], 5)
    assert got["all_integral"] is False
    assert got["coefficients"] == [Fraction(1, 2 ** (k + 1))
                                   for k in range(5)]


def test_coefficients_stay_exact_far_past_double_range():
    got = rational_gf_coefficients([0, 1], [1, -1, -1], 300)
    fib = [0, 1]
    while len(fib) < 300:
        fib.append(fib[-1] + fib[-2])
    assert got["coefficients"][-1] == fib[-1]
    assert fib[-1] > 2 ** 53


def test_gf_validation():
    with pytest.raises(ValueError, match="constant term"):
        rational_gf_coefficients([1], [0, 1], 5)
    with pytest.raises(ValueError, match="must be positive"):
        rational_gf_coefficients([1], [1, -1], 0)


# --------------------------------------------------------------------
# Dominant singularity
# --------------------------------------------------------------------

def test_the_fibonacci_growth_rate_is_the_golden_ratio():
    out = dominant_singularity_growth([1, -1, -1])
    assert out["growth_rate"] == pytest.approx((1 + math.sqrt(5)) / 2,
                                               abs=1e-12)
    assert out["radius"] == pytest.approx((math.sqrt(5) - 1) / 2, abs=1e-12)


def test_the_measured_ratio_approaches_the_predicted_rate():
    fib = rational_gf_coefficients([0, 1], [1, -1, -1], 40)["coefficients"]
    out = dominant_singularity_growth([1, -1, -1], fib)
    assert out["relative_gap"] < 1e-12


def test_a_simple_pole_is_found_exactly():
    out = dominant_singularity_growth([1, -3])
    assert out["growth_rate"] == pytest.approx(3.0, abs=1e-12)


def test_a_denominator_with_no_positive_root_is_refused():
    with pytest.raises(ValueError, match="Pringsheim"):
        dominant_singularity_growth([1, 0, 1])   # roots at +-i


# --------------------------------------------------------------------
# Transfer theorem
# --------------------------------------------------------------------

def test_the_transfer_ratio_converges_to_one():
    ratios = [abs(singularity_transfer(0.5, n)["ratio"] - 1)
              for n in (10, 100, 1000)]
    assert ratios[0] > ratios[1] > ratios[2]
    assert ratios[2] < 2e-4


def test_the_first_order_correction_earns_its_name():
    # with the 1 + a(a-1)/(2n) factor the gap shrinks like n^-2, not n^-1
    for n in (10, 100):
        t = singularity_transfer(0.5, n)
        assert abs(t["corrected_ratio"] - 1) < abs(t["ratio"] - 1) / 50


def test_integer_alpha_reduces_to_binomials():
    # (1-x)^-2 has coefficients n+1
    for n in (1, 5, 20):
        assert singularity_transfer(2, n)["exact_coefficient"] == \
            pytest.approx(n + 1)
    # (1-x)^-1 is all ones
    assert singularity_transfer(1, 50)["exact_coefficient"] == \
        pytest.approx(1.0)


def test_catalan_asymptotics_derive_from_the_transfer_theorem():
    # C_n = binom(2n,n)/(n+1) ~ 4^n / (sqrt(pi) n^(3/2)); the n^(-3/2)
    # is alpha = -1/2 of the sqrt singularity of the Catalan GF
    for n in (50, 200):
        exact = catalan_number(n)
        asym = 4.0 ** n / (math.sqrt(math.pi) * n ** 1.5)
        ratio = asym / float(exact)
        assert ratio == pytest.approx(1.0, abs=5.0 / n)


def test_transfer_validation():
    with pytest.raises(ValueError, match="non-positive integer"):
        singularity_transfer(0, 10)
    with pytest.raises(ValueError, match="n must be positive"):
        singularity_transfer(0.5, 0)


# --------------------------------------------------------------------
# Stirling's series
# --------------------------------------------------------------------

def test_the_error_bound_holds_at_every_n_and_term_count():
    for n in (1, 2, 5, 10, 50, 170, 1000):
        for k in range(5):
            assert stirling_series_error(n, k)["error_within_bound"] is True


def test_more_terms_help_until_the_double_floor():
    errs = [stirling_series_error(10, k)["error"] for k in range(5)]
    assert all(a > b for a, b in zip(errs[:4], errs[1:4]))


def test_the_bound_is_the_first_omitted_term():
    out = stirling_series_error(10, 2)
    b6 = Fraction(1, 42)
    assert out["bound"] == pytest.approx(float(b6) / (6 * 5 * 10 ** 5))


def test_the_series_is_sharp_not_just_valid():
    # the achieved error should be MOST of the bound, since the
    # envelope is tight for an alternating-enveloping series
    out = stirling_series_error(5, 3)
    assert out["error"] > 0.5 * out["bound"]


def test_stirling_validation():
    with pytest.raises(ValueError, match="must be positive"):
        stirling_series_error(0)
    with pytest.raises(ValueError, match="B12"):
        stirling_series_error(10, 5)


# --------------------------------------------------------------------
# Derangements
# --------------------------------------------------------------------

def test_the_recurrence_agrees_with_the_enumerative_shelf():
    for n in range(11):
        assert derangement_rounding(n)["derangements"] == \
            derangements(n)


def test_the_rounding_identity_holds_exactly_for_every_n_to_60():
    for n in range(1, 61):
        out = derangement_rounding(n)
        assert out["is_nearest_integer"] is True
        assert out["within_theoretical_bound"] is True


def test_the_distance_bound_shrinks_like_one_over_n():
    d10 = derangement_rounding(10)["distance_bound"]
    d40 = derangement_rounding(40)["distance_bound"]
    assert d40 < d10
    assert d10 < 1.0 / 11


def test_derangements_far_past_double_range_are_exact():
    out = derangement_rounding(50)
    assert out["exact"] == ("1118871961078248050463025807075773432401"
                            "1354208865721592720336801")
    assert out["derangements"] > 2 ** 53


def test_derangement_validation():
    with pytest.raises(ValueError, match="non-negative"):
        derangement_rounding(-1)


# --------------------------------------------------------------------
# Hardy-Ramanujan
# --------------------------------------------------------------------

def test_exact_partition_counts_agree_with_the_enumerative_shelf():
    for n in (10, 50, 100):
        assert hardy_ramanujan_partitions(n)["partitions"] == \
            partition_count(n)


def test_the_relative_error_decays_but_slowly():
    errs = [hardy_ramanujan_partitions(n)["relative_error"]
            for n in (10, 100, 1000)]
    assert all(e > 0 for e in errs)          # always an overestimate
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] > 0.04                     # still 4.6% off at n = 100


def test_known_partition_values():
    assert hardy_ramanujan_partitions(100)["partitions"] == 190569292
    assert hardy_ramanujan_partitions(1000)["exact"] == \
        "24061467864032622473692149727991"


def test_hardy_ramanujan_validation():
    with pytest.raises(ValueError, match="must be positive"):
        hardy_ramanujan_partitions(0)
