"""Tests for otmtest.ot_mmd_two_sample.

The estimators are checked against a direct transcription of Gretton et
al. (2012), and the test against its own operating characteristics.
"""

import numpy as np
import pytest

from morie.fn.otmtest import _gram, ot_mmd_two_sample


def _brute_mmd2(A, B, gamma, unbiased):
    """Equations from the paper, written out with explicit loops."""
    m, n = len(A), len(B)
    kxx = sum(np.exp(-gamma * np.sum((A[i] - A[j]) ** 2)) for i in range(m) for j in range(m) if not (unbiased and i == j))
    kyy = sum(np.exp(-gamma * np.sum((B[i] - B[j]) ** 2)) for i in range(n) for j in range(n) if not (unbiased and i == j))
    kxy = sum(np.exp(-gamma * np.sum((A[i] - B[j]) ** 2)) for i in range(m) for j in range(n))
    if unbiased:
        return kxx / (m * (m - 1)) + kyy / (n * (n - 1)) - 2 * kxy / (m * n)
    return kxx / m**2 + kyy / n**2 - 2 * kxy / (m * n)


@pytest.mark.parametrize("unbiased", [False, True])
def test_matches_the_paper_formulas(unbiased):
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (12, 2))
    B = rng.normal(0.4, 1, (9, 2))
    res = ot_mmd_two_sample(A, B, gamma=0.5, B=1, unbiased=unbiased, seed=1)
    assert res["statistic"] == pytest.approx(_brute_mmd2(A, B, 0.5, unbiased), rel=1e-12)


def test_biased_estimate_is_non_negative():
    """MMD_b^2 is a squared RKHS norm, so it cannot go below zero."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        res = ot_mmd_two_sample(rng.normal(0, 1, 30), rng.normal(0, 1, 30), B=1, seed=1)
        assert res["statistic"] >= -1e-12


def test_same_distribution_is_not_rejected():
    rng = np.random.default_rng(2)
    res = ot_mmd_two_sample(rng.normal(0, 1, 60), rng.normal(0, 1, 60), B=199, seed=3)
    assert res["p_value"] > 0.05


def test_location_shift_is_rejected():
    rng = np.random.default_rng(3)
    res = ot_mmd_two_sample(rng.normal(0, 1, 60), rng.normal(2.0, 1, 60), B=199, seed=3)
    assert res["p_value"] <= 0.01


def test_linear_kernel_is_blind_to_a_mean_preserving_difference():
    """A linear kernel compares means only; rbf sees the scale change.

    This is the practical content of "characteristic": the rbf kernel
    detects any difference in distribution, the linear one does not.
    """
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 200), rng.normal(0, 5, 200)
    lin = ot_mmd_two_sample(a, b, kernel="linear", B=199, seed=3)["p_value"]
    rbf = ot_mmd_two_sample(a, b, kernel="rbf", B=199, seed=3)["p_value"]
    assert rbf <= 0.01
    assert lin > 0.05


def test_median_heuristic_sets_a_positive_gamma():
    rng = np.random.default_rng(5)
    res = ot_mmd_two_sample(rng.normal(0, 1, 40), rng.normal(0, 1, 40), B=1, seed=1)
    assert res["gamma"] > 0


def test_gram_is_symmetric_and_unit_diagonal_for_rbf():
    Z = np.random.default_rng(6).normal(0, 1, (8, 3))
    K = _gram(Z, Z, "rbf", 0.7)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)


def test_p_value_is_a_rank_and_cannot_be_zero():
    rng = np.random.default_rng(7)
    res = ot_mmd_two_sample(rng.normal(0, 1, 40), rng.normal(3, 1, 40), B=99, seed=3)
    assert res["p_value"] >= 1 / 100


def test_validates_inputs():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, (20, 2))
    with pytest.raises(ValueError, match="share a feature dimension"):
        ot_mmd_two_sample(a, rng.normal(0, 1, (20, 3)))
    with pytest.raises(ValueError, match="at least 2 observations"):
        ot_mmd_two_sample(a[:1], a)
    with pytest.raises(ValueError, match="kernel must be one of"):
        ot_mmd_two_sample(a, a, kernel="cosine")
    with pytest.raises(ValueError, match="gamma must be positive"):
        ot_mmd_two_sample(a, a, gamma=0)
    with pytest.raises(ValueError, match="B must be at least 1"):
        ot_mmd_two_sample(a, a, B=0)
    with pytest.raises(ValueError, match="must be finite"):
        ot_mmd_two_sample(np.array([[1.0, np.nan], [2.0, 3.0]]), a)
