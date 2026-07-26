"""rng141: Wiener MSE cost function (Rangayyan 2024, Eq. 3.166, p. 175).

The cost is pinned against the book's own optimum rather than a transcribed
constant: Eq. (3.169) gives w_o = Phi^-1 Theta and Eq. (3.172) gives
J_min = sigma_d^2 - Theta' Phi^-1 Theta, so J(w_o) must equal J_min and no
other w may beat it.
"""

import numpy as np
import pytest

from morie.fn.rng141 import rangayyan_ch3_mse_cost_function as mse
from morie.fn.rng143 import rangayyan_ch3_autocorrelation_matrix as phi_mat


def _problem(seed=17, M=4):
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.standard_normal(3000), np.ones(4) / 4, mode="same")
    Phi = phi_mat(x, M)["array"]
    Theta = rng.standard_normal(M)
    sigma_d = 2.0
    return Phi, Theta, sigma_d


def test_rng141_at_the_optimum_equals_J_min():
    """Eq. (3.166) evaluated at Eq. (3.169) must give Eq. (3.172)."""
    Phi, Theta, sd = _problem()
    w_o = np.linalg.solve(Phi, Theta)
    J_min = sd**2 - Theta @ np.linalg.solve(Phi, Theta)
    assert mse(w_o, Theta, Phi, sd)["value"] == pytest.approx(J_min, rel=1e-12)


def test_rng141_optimum_is_a_true_minimum():
    """No perturbation of w_o may lower J -- it is a second-order function."""
    Phi, Theta, sd = _problem()
    w_o = np.linalg.solve(Phi, Theta)
    J_o = mse(w_o, Theta, Phi, sd)["value"]
    rng = np.random.default_rng(123)
    for _ in range(50):
        w = w_o + rng.standard_normal(w_o.size) * 0.1
        assert mse(w, Theta, Phi, sd)["value"] > J_o


def test_rng141_zero_weights_give_the_desired_variance():
    """J(0) = sigma_d^2: with no filter, the error IS the desired response."""
    Phi, Theta, sd = _problem()
    assert mse(np.zeros(Theta.size), Theta, Phi, sd)["value"] == pytest.approx(sd**2)


def test_rng141_matches_the_direct_expectation_of_the_squared_error():
    """Eq. (3.159) is E[e^2(n)]; compute that from data and compare.

    This is the check that would catch a sign error in the cross-terms, which
    the quadratic-form check alone cannot -- flipping both -w'Theta and
    -Theta'w still yields a paraboloid, just one with the wrong minimum.
    """
    rng = np.random.default_rng(31)
    N, M = 40_000, 3
    x = rng.standard_normal(N)
    d = np.convolve(x, [0.5, -0.2, 0.1], mode="full")[:N] + rng.standard_normal(N) * 0.1
    X = np.column_stack([x[M - 1 - k : N - k] for k in range(M)])
    dd = d[M - 1 :]
    Phi = X.T @ X / X.shape[0]
    Theta = X.T @ dd / X.shape[0]
    sigma_d = float(np.sqrt(np.mean(dd**2)))
    w = np.array([0.4, -0.1, 0.05])
    empirical = float(np.mean((dd - X @ w) ** 2))
    assert mse(w, Theta, Phi, sigma_d)["value"] == pytest.approx(empirical, rel=1e-10)


def test_rng141_rejects_shape_mismatch():
    Phi, Theta, sd = _problem()
    with pytest.raises(ValueError, match="same length"):
        mse(np.zeros(3), Theta, Phi, sd)
    with pytest.raises(ValueError, match="must have shape"):
        mse(np.zeros(4), Theta, np.eye(3), sd)


def test_rng141_rejects_negative_sigma_d():
    Phi, Theta, _ = _problem()
    with pytest.raises(ValueError, match="non-negative"):
        mse(np.zeros(4), Theta, Phi, -1.0)
