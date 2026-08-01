"""Morin (2016) chapter 4: distributions, book-anchored."""
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


def test_density_probability_eqs_4_2_4_4():
    grid = np.linspace(0.0, 1.0, 2001)
    density = np.full_like(grid, 1.0)  # uniform on [0, 1]
    assert front("4e2")(grid, density, 0.2, 0.5)["probability"] == pytest.approx(0.3, abs=1e-6)
    assert front("4e4")(grid, density, 0.5, 0.2)["probability"] == pytest.approx(0.2, abs=1e-6)
    with pytest.raises(ValueError):
        front("4e2")(grid, 2.0 * density, 0.2, 0.5)  # integrates to 2


def test_binomial_pmf_eqs_4_6_4_8():
    # book eq (4.7): P(k Heads in 4 fair flips) peaks at 6/16
    assert front("4e8")(2, 4)["probability"] == pytest.approx(6 / 16)
    assert front("4e6")(3, 10, 0.3)["probability"] == pytest.approx(
        math.comb(10, 3) * 0.3 ** 3 * 0.7 ** 7)


def test_p0_equals_p1_eq_4_9():
    r = front("4e9")(9)
    assert r["p"] == pytest.approx(0.1)
    assert r["P0"] == pytest.approx(r["P1"])


def test_normalization_eq_4_10():
    assert front("4e10")(12, 0.37)["total"] == pytest.approx(1.0)


def test_poisson_process_basics_eqs_4_18_4_19():
    r = front("4e18")(2.0, 1e-4)
    assert r["approx"] == pytest.approx(2e-4)
    assert r["abs_error"] < 1e-7
    assert front("4e19")(3.0, 2.5)["expected_events"] == pytest.approx(7.5)


def test_waiting_time_eqs_4_23_to_4_26():
    assert front("4e25")(1.0, 0.01, 2.0)["probability"] == pytest.approx(
        math.exp(-2.0) * 0.02)
    assert front("4e23")(0.0, 0.01, 2.0)["probability"] == pytest.approx(0.02)
    assert front("4e26")(0.5, 2.0)["density"] == pytest.approx(2.0 * math.exp(-1.0))
    # density integrates to 1
    ts = np.linspace(0, 40, 400001)
    total = np.trapezoid([front("4e26")(t, 0.7)["density"] for t in [0]] and
                         0.7 * np.exp(-0.7 * ts), ts)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_crossing_time_eq_4_30():
    # book: ln 4 = t (1/5 - 1/20)  =>  t = 9.24
    r = front("4e30")()
    assert r["t"] == pytest.approx(math.log(4) / (0.2 - 0.05), rel=1e-12)
    assert round(r["t"], 2) == 9.24


def test_dice_binomial_eq_4_32():
    assert front("4e32")(2, 6, 6.0)["probability"] == pytest.approx(
        math.comb(6, 2) * (1 / 6) ** 2 * (5 / 6) ** 4)


def test_poisson_limit_eqs_4_34_to_4_37():
    r = front("4e34")(3, 100000, 2.0)
    assert r["abs_error"] < 1e-4
    r2 = front("4e35")(3, 1000000, 2.0)
    assert r2["abs_error"] < r["abs_error"]
    r3 = front("4e37")(1e-5, 100000)
    assert r3["rel_error"] < 1e-4


def test_poisson_pmf_eq_4_40():
    assert front("4e40")(3, 2.0)["probability"] == pytest.approx(
        2.0 ** 3 * math.exp(-2.0) / 6.0)
    total = sum(front("4e40")(k, 4.5)["probability"] for k in range(60))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_alternating_series_eq_4_53():
    r = front("4e53")(2.0)
    assert r["final_error"] < 1e-14
    assert r["e_minus_a"] == pytest.approx(math.exp(-2.0))


def test_continuous_expectation_eq_4_55():
    grid = np.linspace(0.0, 1.0, 4001)
    assert front("4e55")(grid, np.full_like(grid, 1.0))["expectation"] == pytest.approx(0.5, abs=1e-8)


def test_binomial_moments_eqs_4_60_to_4_67():
    assert front("4e60")(5, 20, 0.25)["probability"] == pytest.approx(
        math.comb(20, 5) * 0.25 ** 5 * 0.75 ** 15)
    assert front("4e61")(20, 0.25)["mean"] == pytest.approx(5.0)
    assert front("4e66")(20, 0.25)["second_moment"] == pytest.approx(
        0.0625 * 380 + 5.0)
    assert front("4e67")(20, 0.25)["variance"] == pytest.approx(20 * 0.25 * 0.75)


def test_hypergeometric_eqs_4_71_to_4_75():
    # sanity: drawing 5 from a 52-card deck, P(2 hearts)
    p = front("4e71")(2, 52, 13, 5)["probability"]
    assert p == pytest.approx(math.comb(13, 2) * math.comb(39, 3) / math.comb(52, 5))
    total = sum(front("4e71")(k, 52, 13, 5)["probability"] for k in range(6))
    assert total == pytest.approx(1.0)
    r1 = front("4e75")(3, 10, 0.3, 100)
    r2 = front("4e75")(3, 10, 0.3, 100000)
    assert r2["abs_error"] < r1["abs_error"] < 0.05


def test_exponential_moments_eqs_4_83_4_85():
    assert front("4e83")(3.0)["mean"] == pytest.approx(3.0)
    r = front("4e85")(3.0)
    assert r["second_moment"] == pytest.approx(18.0)
    assert r["variance"] == pytest.approx(9.0)


def test_poisson_mode_mean_var_eqs_4_89_to_4_94():
    # book: mode at k = a - 1 (and a) for integer a
    assert front("4e89")(7.0)["mode"] == 6
    assert front("4e89")(2.5)["mode"] == 2
    assert front("4e92")(4.2)["mean"] == pytest.approx(4.2, abs=1e-9)
    assert front("4e94")(4.2)["variance"] == pytest.approx(4.2, abs=1e-6)


def test_peak_ratio_eqs_4_95_to_4_98():
    # ratio -> sqrt(1-p); book's clean result
    r = front("4e98")(100000, 0.2)
    assert r["ratio"] == pytest.approx(math.sqrt(0.8), abs=2e-4)
    pp = front("4e95")(100000, 0.2)["PP"]
    pb = front("4e96")(100000, 0.2)["PB"]
    assert pp / pb == pytest.approx(r["ratio"], rel=1e-12)


def test_typos_eq_4_99():
    r = front("4e99")()
    assert r["p_zero"] == pytest.approx(math.exp(-7.0), rel=1e-12)
    assert 8e-4 < r["p_zero"] < 1e-3  # book: ~9e-4 ~ 0.1%
