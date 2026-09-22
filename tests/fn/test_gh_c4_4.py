"""Tests for gh_c4_4.ghosal_dp_cov."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_4 import ghosal_dp_cov


def test_gh_c4_4_basic():
    """Test basic functionality: covariance from a single overlap measure with three dependences."""
    # Each input must be a scalar (single-element array) representing the
    # mutual dependence of subsets A, B, and A∩B under G0.
    G0_AB = np.array([0.60])  # P(A ∩ B) under null
    G0_A = np.array([0.50])  # P(A) under null
    G0_B = np.array([0.40])  # P(B) under null
    alpha = np.array([3.0])  # 1 + |alpha| = 4 enters the denominator

    result = ghosal_dp_cov(G0_AB, G0_A, G0_B, alpha)

    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))

    # Independent recomputation from eq. 4.4:
    # cov(P(A), P(B)) = (G0(A ∩ B) - G0(A) G0(B)) / (1 + |alpha|)
    expected = (0.60 - 0.50 * 0.40) / (1.0 + 3.0)
    assert np.allclose(est, expected)


def test_gh_c4_4_edge():
    """Test edge cases: dependence-free inputs and zero alpha."""
    # If A and B are independent under G0, the estimated covariance must be 0.
    G0_AB = np.array([0.20])
    G0_A = np.array([0.50])
    G0_B = np.array([0.40])  # 0.50 * 0.40 == 0.20
    alpha = np.array([5.0])

    result = ghosal_dp_cov(G0_AB, G0_A, G0_B, alpha)

    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.allclose(est, 0.0)

    # With alpha = 0 the denominator is 1, so estimate == raw plug-in covariance.
    G0_AB2 = np.array([0.35])
    G0_A2 = np.array([0.50])
    G0_B2 = np.array([0.40])
    alpha2 = np.array([0.0])
    res2 = ghosal_dp_cov(G0_AB2, G0_A2, G0_B2, alpha2)
    est2 = np.asarray(res2["estimate"], dtype=float)
    expected2 = 0.35 - 0.50 * 0.40
    assert np.allclose(est2, expected2)
