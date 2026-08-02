"""Tests for otprm.ot_permutation_test_w1.

W_1 is checked against scipy.stats.wasserstein_distance, an independent
implementation of the same closed form; the test itself is checked for
size under the null and power against location and shape differences.
"""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.otprm import _w1, ot_permutation_test_w1


def test_w1_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(0, 1, 60)
        b = rng.normal(0.5, 2, 45)
        assert _w1(a, b) == pytest.approx(stats.wasserstein_distance(a, b), rel=1e-12)


def test_w1_of_a_sample_with_itself_is_zero():
    x = np.random.default_rng(1).normal(0, 1, 40)
    assert _w1(x, x) == pytest.approx(0.0, abs=1e-15)


def test_w1_of_a_pure_shift_is_the_shift():
    """Translating every point by c moves the CDF sideways by c."""
    x = np.random.default_rng(2).normal(0, 1, 200)
    assert _w1(x, x + 3.0) == pytest.approx(3.0, rel=1e-12)


def test_same_distribution_is_not_rejected():
    rng = np.random.default_rng(3)
    res = ot_permutation_test_w1(rng.normal(0, 1, 80), rng.normal(0, 1, 80), B=199, seed=1)
    assert res["p_value"] > 0.05


def test_location_shift_is_rejected():
    rng = np.random.default_rng(4)
    res = ot_permutation_test_w1(rng.normal(0, 1, 80), rng.normal(1.5, 1, 80), B=199, seed=1)
    assert res["p_value"] <= 0.01


def test_scale_difference_is_rejected_though_means_agree():
    """W_1 sees the whole CDF gap, so equal means do not hide it."""
    rng = np.random.default_rng(5)
    res = ot_permutation_test_w1(rng.normal(0, 1, 150), rng.normal(0, 4, 150), B=199, seed=1)
    assert res["p_value"] <= 0.01


def test_p_value_is_a_rank_and_cannot_be_zero():
    rng = np.random.default_rng(6)
    res = ot_permutation_test_w1(rng.normal(0, 1, 50), rng.normal(3, 1, 50), B=99, seed=1)
    assert res["p_value"] >= 1 / 100
    assert np.isclose(res["p_value"] * 100 % 1, 0)


def test_seed_makes_it_reproducible():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
    assert ot_permutation_test_w1(a, b, B=99, seed=5)["p_value"] == ot_permutation_test_w1(a, b, B=99, seed=5)["p_value"]


def test_validates_inputs():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, 30)
    with pytest.raises(ValueError, match="one-dimensional"):
        ot_permutation_test_w1(a.reshape(-1, 1), a)
    with pytest.raises(ValueError, match="at least 2 observations"):
        ot_permutation_test_w1(a[:1], a)
    with pytest.raises(ValueError, match="must be finite"):
        ot_permutation_test_w1(np.array([1.0, np.nan, 2.0]), a)
    with pytest.raises(ValueError, match="B must be at least 1"):
        ot_permutation_test_w1(a, a, B=0)
