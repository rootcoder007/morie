"""Morin (2016) chapters 5-7 (+ two misnamed modules), book-anchored."""
import math

import numpy as np
import pytest

from morie.fn import _morin

P = "david_j_morin_probability_for_the_enthusiastic_beginner"


def front(suffix):
    import importlib
    mod = importlib.import_module(f"morie.fn.{P}{suffix}")
    ch, e = suffix.split("e")
    return getattr(mod, f"{P}_chapter_{ch}_equation_{e}")


# ------------------------------------------------------------- chapter 5

def test_centered_binomial_eqs_5_3_5_5():
    # 2n = 100 flips: PB(0) is the n=50 exact value 0.07959
    assert front("5e3")(0, 50)["probability"] == pytest.approx(0.07959, abs=5e-6)
    assert front("5e5")(3, 50)["probability"] == pytest.approx(
        front("5e3")(3, 50)["probability"], rel=1e-12)


def test_gaussian_approx_eqs_5_4_5_13_5_14():
    # PG(0) = 1/sqrt(pi 50) = 0.07979 (book)
    assert front("5e13")(0, 50)["PG"] == pytest.approx(0.07979, abs=5e-6)
    assert front("5e4")(0, 50)["rel_error"] < 0.003
    # n-flip form at even deviation matches the 2n form with n -> n/2
    assert front("5e14")(2.0, 100)["PG"] == pytest.approx(
        front("5e13")(2.0, 50)["PG"], rel=1e-12)
    # Gaussian tracks the exact binomial within 1% near the peak at n=200 flips
    for x in range(0, 6):
        exact = front("5e3")(x, 100)["probability"]
        approx = front("5e13")(float(x), 100)["PG"]
        assert abs(approx - exact) / exact < 0.01


def test_biased_gaussian_eq_5_15():
    n, p = 10000, 0.3
    exact = _morin.binomial_pmf(int(p * n), n, p)
    assert front("5e15")(0.0, n, p)["PG"] == pytest.approx(exact, rel=2e-3)


def test_poisson_stirling_and_gaussian_eqs_5_16_5_23():
    assert front("5e16")(400, 400.0)["rel_error"] < 1e-3
    r = front("5e17")(0.0, 400.0)
    assert r["approx"] == pytest.approx(r["exact"], rel=1e-3)
    # Gaussian limit of Poisson at large a
    a = 400.0
    for k in (380, 400, 420):
        assert front("5e23")(k, a)["PG"] == pytest.approx(
            _morin.poisson_pmf(k, a), rel=0.02)


def test_tail_eq_5_25():
    r = front("5e25")()
    # book: e^(-200)/sqrt(2 pi) ~ 1e-87
    assert r["area_fraction"] == pytest.approx(math.exp(-200) / math.sqrt(2 * math.pi), rel=1e-9)
    assert 1e-88 < r["area_fraction"] < 1e-86


def test_expected_count_gaussian_eq_5_28():
    # book: 100,000 trials, mu=35, sigma=5.4; peak count = 100000/sqrt(2 pi 5.4^2)
    r = front("5e28")(35.0)
    assert r["expected_count"] == pytest.approx(
        100000 / math.sqrt(2 * math.pi * 5.4 ** 2), rel=1e-12)


def test_pmf_sd_eq_5_31():
    # book: values 2, 3.2, 7 with probs .6, .1, .3 -> mean 3.62, sd 2.24
    r = front("5e31")([2.0, 3.2, 7.0], [0.6, 0.1, 0.3])
    assert r["mean"] == pytest.approx(3.62)
    assert round(r["sd"], 2) == 2.24
    # and sigma_avg over 100 draws = 0.224 (book, below eq (5.31))
    assert round(r["sd"] / 10, 3) == 0.224


# ------------------------------------------------------------- chapter 6

X5 = [2.0, 3.0, 3.0, 5.0, 7.0]
Y5 = [1.0, 1.0, 3.0, 4.0, 6.0]


def test_linear_model_eqs_6_3_to_6_6():
    r = front("6e5")(2.0, 3.0, 4.0)
    assert r["sigma_y"] == pytest.approx(math.sqrt(36 + 16))
    assert front("6e4")(2.0, 1.5, 0.5)["mu_y"] == pytest.approx(3.5)
    rr = front("6e6")(2.0, 3.0, 4.0)
    assert rr["r"] == pytest.approx(6.0 / math.sqrt(52))
    # limiting cases (book below eq (6.5)): sigma_z = 0 -> r = 1; m = 0 -> r = 0
    assert front("6e6")(2.0, 3.0, 0.0)["r"] == pytest.approx(1.0)
    assert front("6e6")(0.0, 3.0, 4.0)["r"] == pytest.approx(0.0)


def test_worked_conversion_eq_6_17():
    # book: m = 0.4, sigma_x = 1, sigma_z = 1 -> sigma_y = 1.08, r = 0.37
    r = front("6e17")(0.4, 1.0, 1.0)
    assert round(r["sigma_y"], 2) == 1.08
    assert round(r["r"], 2) == 0.37


def test_covariance_and_r_eqs_6_8_to_6_14():
    x = np.array(X5) - np.mean(X5)
    y = np.array(Y5) - np.mean(Y5)
    assert front("6e8")(x, y)["cov"] == pytest.approx(_morin.sample_cov(X5, Y5))
    assert front("6e14")(X5, Y5)["cov"] == pytest.approx(15.2 - 4.0 * 3.0)
    assert front("6e9")(X5, Y5)["r"] == pytest.approx(_morin.sample_r(X5, Y5))
    assert front("6e12")(X5, Y5)["r"] == pytest.approx(front("6e55")(X5, Y5)["r"])
    # book problem 6.9: slope = 1 for this dataset
    assert front("6e13")(X5, Y5)["slope"] == pytest.approx(1.0)


def test_best_predictor_eqs_6_22_6_23():
    y = [1.0, 4.0, 4.0, 7.0]
    assert front("6e22")(y)["best_prediction"] == pytest.approx(4.0)
    assert front("6e23")(y)["best_prediction"] == pytest.approx(4.0)


def test_improvement_eq_6_27():
    assert front("6e27")(0.6)["mse_fraction_remaining"] == pytest.approx(0.64)
    assert front("6e27")(-0.6)["mse_fraction_remaining"] == pytest.approx(0.64)


def test_reverse_and_retest_eqs_6_36_to_6_40():
    assert front("6e36")(0.5, 2.0, 4.0)["slope"] == pytest.approx(0.25)
    # equal signal and noise: r = 1/sqrt(2) ~ 0.71 (book eq (6.38))
    assert front("6e37")(10.6, 10.6)["r"] == pytest.approx(1 / math.sqrt(2))
    assert round(front("6e38")()["r"], 2) == 0.71
    assert front("6e38")()["sigma_y"] == pytest.approx(15.0, rel=1e-9)
    assert front("6e39")(0.5, 8.0)["yavg"] == pytest.approx(4.0)
    assert front("6e40")(1 / math.sqrt(2), 10.0)["yavg"] == pytest.approx(5.0)


def test_least_squares_worked_dataset_eqs_6_42_to_6_50():
    # book problem 6.9: <x2> = 19.2, <xy> = 15.2, A = 1, B = -1
    assert np.mean(np.array(X5) ** 2) == pytest.approx(19.2)
    assert np.mean(np.array(X5) * np.array(Y5)) == pytest.approx(15.2)
    r = front("6e47")(X5, Y5)
    assert r["A"] == pytest.approx(1.0)
    assert r["B"] == pytest.approx(-1.0)
    assert front("6e46")(X5, Y5)["A"] == pytest.approx(1.0)
    assert front("6e42")(X5, Y5)["S"] == pytest.approx(
        float(np.sum((np.array(Y5) - (np.array(X5) - 1.0)) ** 2)))
    assert front("6e43")(X5, Y5)["S"] == pytest.approx(front("6e42")(X5, Y5)["S"])
    C = front("6e50")(X5, Y5)["C"]
    assert C == pytest.approx(_morin.least_squares_fit(Y5, X5)[0])


def test_slope_product_and_r_eqs_6_53_6_55():
    r = front("6e53")(X5, Y5)
    assert r["slope_product_AC"] == pytest.approx(r["r"] ** 2)
    assert front("6e55")(X5, Y5)["r"] == pytest.approx(_morin.sample_r(X5, Y5))


def test_independence_zero_cov_eq_6_63():
    x = [-1.0, -1.0, 1.0, 1.0]
    y = [-1.0, 1.0, -1.0, 1.0]
    r = front("6e63")(x, y)
    assert r["near_zero"]


def test_continuous_independence_and_sum_eqs_6_64_to_6_70():
    g = np.linspace(-16.0, 16.0, 3201)
    dx = np.exp(-g ** 2 / 2) / math.sqrt(2 * math.pi)
    dy = np.exp(-g ** 2 / (2 * 4)) / math.sqrt(2 * math.pi * 4)
    assert front("6e64")(g, dx, g, dy)["total_mass"] == pytest.approx(1.0, abs=1e-6)
    # convolution of N(0,1) and N(0,4) at z equals N(0,5) density -- eq (6.70)
    for z in (0.0, 1.0, 2.5):
        num = front("6e65")(g, dx, g, dy, z)["density"]
        closed = front("6e70")(z, 1.0, 2.0)["density"]
        assert num == pytest.approx(closed, rel=1e-4)
    r = front("6e66")(g, dx, g, dy, 1.0, 0.01)
    assert r["probability"] == pytest.approx(0.01 * front("6e70")(1.0, 1.0, 2.0)["density"], rel=1e-4)


def test_strip_and_worked_sigma_eqs_6_74_6_76():
    assert front("6e74")(0.5, 2.0, 4.0, 8.0)["x"] == pytest.approx(2.0)
    r = front("6e76")()
    assert round(r["sigma_y"], 0) == 13.0  # book: sqrt(7.5^2 + 10.6^2) = 13
    assert r["sigma_y"] == pytest.approx(math.sqrt(7.5 ** 2 + 10.6 ** 2), rel=1e-12)


def test_excess_factor_eq_6_81():
    assert front("6e81")(0.5)["factor"] == pytest.approx(math.sqrt(1/3))
    assert front("6e81")(0.0)["factor"] == pytest.approx(1.0)


def test_intercept_forms_and_means_eqs_6_82_6_83_6_92():
    assert front("6e82")(X5, Y5)["B"] == pytest.approx(-1.0)
    r = front("6e83")()
    assert r["xbar"] == pytest.approx(4.0) and r["ybar"] == pytest.approx(3.0)
    assert abs(front("6e92")(X5, Y5)["residual_sum"]) < 1e-12


def test_misnamed_modules_alias_correct_equations():
    # 10e6: the "10.6" is a number inside eq (6.76); module = sigma_y of the model
    r = front("10e6")()
    assert r["sigma_y"] == pytest.approx(math.sqrt(7.5 ** 2 + 10.6 ** 2))
    # 19e2: the "19.2" is <x^2> inside eq (6.89); module = intercept B = -1
    r2 = front("19e2")()
    assert r2["B"] == pytest.approx(-1.0)
    assert r2["A"] == pytest.approx(1.0)


# ------------------------------------------------------------- chapter 7

def test_domain_check_eq_7_5():
    assert front("7e5")(1.0, 10000)["well_inside"]
    assert not front("7e5")(50.0, 100)["well_inside"]


def test_exp_series_eqs_7_7_to_7_11():
    r = front("7e7")(1.5)
    assert r["final_error"] < 1e-12
    assert front("7e9")(0.01)["abs_error"] < 1e-4
    assert front("7e10")(3.0)["error"] < 1e-10
    assert front("7e11")(4.2)["error"] < 1e-12


def test_one_plus_a_ladder_eqs_7_14_to_7_24():
    # book worked case: (1 - 1/N)^n with N=365, n=23
    r = front("7e14")(-1/365, 23)
    assert r["exact"] == pytest.approx((1 - 1/365) ** 23, rel=1e-12)
    assert abs(r["approx"] - r["exact"]) / r["exact"] < 1e-3
    r21 = front("7e21")(0.05, 100)
    assert r21["rel_error"] < 1e-10
    assert front("7e23")(0.001, 100)["valid"]
    assert not front("7e23")(0.1, 1000)["valid"]
    # second order beats first order when na^2 is not tiny
    a, n = 0.05, 200
    e1 = abs(front("7e23")(a, n)["approx"] - (1 + a) ** n)
    e2 = abs(front("7e24")(a, n)["approx"] - (1 + a) ** n)
    assert e2 < e1


def test_derivative_quotients_eqs_7_31_to_7_35():
    r = front("7e31")(3.0, 1e-6)
    assert r["quotient"] == pytest.approx(6.0, abs=1e-5)
    r33 = front("7e33")(2.0, 5, 1e-7)
    assert r33["quotient"] == pytest.approx(80.0, abs=1e-4)
    r35 = front("7e35")(2.0, 5, 0.1)
    assert r35["abs_error"] < 1e-12
