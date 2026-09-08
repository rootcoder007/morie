"""fzwlc: smoothed Wilcoxon signed-rank test.

Fauzi, R. R. & Maesono, Y. (2023), *Statistical Inference Based on Kernel
Distribution Function Estimators*, Ch. 5 -- in the library.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.fzwlc import fauzi_smoothed_wilcoxon as sw


def test_fzwlc_centred_data_is_not_significant():
    rng = np.random.default_rng(2701)
    assert sw(rng.normal(0.0, 1.0, 200), theta0=0.0)["p_value"] > 0.05


def test_fzwlc_a_clear_location_shift_is_detected():
    rng = np.random.default_rng(2707)
    assert sw(rng.normal(1.0, 1.0, 200), theta0=0.0)["p_value"] < 1e-6


def test_fzwlc_theta0_shifts_the_null_not_the_data():
    """Testing median = 5 on data centred at 5 must NOT reject, even though
    testing median = 0 on the same data does."""
    rng = np.random.default_rng(2711)
    x = rng.normal(5.0, 1.0, 200)
    assert sw(x, theta0=5.0)["p_value"] > 0.05
    assert sw(x, theta0=0.0)["p_value"] < 1e-6


def test_fzwlc_is_equivariant_under_a_common_shift():
    """Shifting both the data and the null hypothesis by the same amount
    cannot change the result -- the test is about the difference."""
    rng = np.random.default_rng(2713)
    x = rng.normal(0.3, 1.0, 150)
    a = sw(x, theta0=0.0)
    b = sw(x + 10.0, theta0=10.0)
    assert a["z"] == pytest.approx(b["z"], rel=1e-6)


def test_fzwlc_direction_shows_in_the_sign_of_z():
    rng = np.random.default_rng(2719)
    up = sw(rng.normal(1.0, 1.0, 200), theta0=0.0)["z"]
    down = sw(rng.normal(-1.0, 1.0, 200), theta0=0.0)["z"]
    assert np.sign(up) != np.sign(down)


def test_fzwlc_one_sided_alternatives_split_the_two_sided_p():
    """A one-sided p in the direction of the effect is roughly half the
    two-sided one, and the opposite side is near 1."""
    rng = np.random.default_rng(2729)
    x = rng.normal(0.5, 1.0, 200)
    two = sw(x, theta0=0.0, alternative="two-sided")["p_value"]
    greater = sw(x, theta0=0.0, alternative="greater")["p_value"]
    less = sw(x, theta0=0.0, alternative="less")["p_value"]
    assert greater == pytest.approx(two / 2, rel=0.15)
    assert less > 0.9


def test_fzwlc_smoothing_bandwidth_is_reported_and_matters():
    """h is the kernel bandwidth that makes this the SMOOTHED test rather
    than the classical one; it must be reported and must move the answer."""
    rng = np.random.default_rng(2731)
    x = rng.normal(0.3, 1.0, 120)
    a = sw(x, theta0=0.0, h=0.05)
    b = sw(x, theta0=0.0, h=1.5)
    assert a["h"] == pytest.approx(0.05)
    assert b["h"] == pytest.approx(1.5)
    assert a["statistic"] != pytest.approx(b["statistic"], rel=1e-9)
