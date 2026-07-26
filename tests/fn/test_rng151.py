"""rng151: optimal Wiener filter for noise removal (Rangayyan 2024, Eq. 3.183, p. 177)."""

import numpy as np
import pytest

from morie.fn.rng151 import rangayyan_ch3_wiener_optimal_for_noise_removal as wiener_o


def test_rng151_reduces_to_the_wiener_hopf_solution():
    """Eq. (3.183) is Eq. (3.169) with Phi = Phi_d + Phi_eta and Theta = Phi_1d."""
    rng = np.random.default_rng(21)
    M = 4
    A = rng.standard_normal((M, M))
    Phi_d = A @ A.T + np.eye(M)
    B = rng.standard_normal((M, M))
    Phi_eta = B @ B.T + np.eye(M)
    Phi_1d = rng.standard_normal(M)
    got = wiener_o(Phi_d, Phi_eta, Phi_1d)["array"]
    assert got == pytest.approx(np.linalg.solve(Phi_d + Phi_eta, Phi_1d), rel=1e-12)


def test_rng151_noiseless_case_solves_against_the_signal_alone():
    """Phi_eta = 0 leaves w_o = Phi_d^-1 Phi_1d."""
    Phi_d = np.array([[2.0, 0.5], [0.5, 2.0]])
    Phi_1d = np.array([1.0, 0.0])
    got = wiener_o(Phi_d, np.zeros((2, 2)), Phi_1d)["array"]
    assert got == pytest.approx(np.linalg.solve(Phi_d, Phi_1d))


def test_rng151_heavy_noise_shrinks_the_filter():
    """The book: 'the gain of the Wiener filter decreases as the SNR decreases.'

    Scaling Phi_eta up must shrink ||w_o|| monotonically.
    """
    Phi_d = np.array([[2.0, 0.5], [0.5, 2.0]])
    Phi_1d = np.array([1.0, 0.2])
    norms = [
        float(np.linalg.norm(wiener_o(Phi_d, np.eye(2) * s, Phi_1d)["array"]))
        for s in (0.1, 1.0, 10.0, 100.0)
    ]
    assert norms == sorted(norms, reverse=True)


def test_rng151_white_case_matches_the_scalar_frequency_response():
    """With everything diagonal, Eq. (3.183) collapses to Eq. (3.186)'s form
    1/(1 + S_eta/S_d) applied per component."""
    sd, se = 4.0, 1.0
    M = 3
    got = wiener_o(np.eye(M) * sd, np.eye(M) * se, np.ones(M) * sd)["array"]
    assert got == pytest.approx(np.ones(M) * (1.0 / (1.0 + se / sd)))


def test_rng151_rejects_singular_sum():
    with pytest.raises(ValueError, match="singular"):
        wiener_o(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2))


def test_rng151_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match Phi_d shape"):
        wiener_o(np.eye(3), np.eye(2), np.ones(3))
    with pytest.raises(ValueError, match="must have length"):
        wiener_o(np.eye(3), np.eye(3), np.ones(2))
