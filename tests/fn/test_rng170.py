"""rng170: Riccati recursion for Phi^-1 (Rangayyan 2024, Eq. 3.215, p. 188).

Eq. (3.215) exists to avoid an M x M inversion at every step. The way to
check it is therefore to do the inversion it avoids and compare.
"""

import numpy as np
import pytest

from morie.fn.rng167 import rangayyan_ch3_rls_phi_recursion as phi_rec
from morie.fn.rng170 import rangayyan_ch3_rls_inverse_recursion as inv_rec


def test_rng170_agrees_with_explicitly_inverting_the_Phi_recursion():
    """The whole point of the matrix-inversion lemma, checked directly."""
    rng = np.random.default_rng(19)
    M, lam = 4, 0.9
    Phi = np.eye(M) * 2.0
    Pinv = np.linalg.inv(Phi)
    for _ in range(25):
        r = rng.standard_normal(M)
        Phi = phi_rec(Phi, r, lam)["array"]
        Pinv = inv_rec(Pinv, r, lam)["array"]
        assert Pinv == pytest.approx(np.linalg.inv(Phi), rel=1e-8)


def test_rng170_output_is_a_true_inverse():
    """Phi(n) @ Phi^-1(n) = I, accumulated over many steps."""
    rng = np.random.default_rng(23)
    M, lam = 3, 0.95
    Phi = np.eye(M)
    Pinv = np.eye(M)
    for _ in range(50):
        r = rng.standard_normal(M)
        Phi = phi_rec(Phi, r, lam)["array"]
        Pinv = inv_rec(Pinv, r, lam)["array"]
    assert Phi @ Pinv == pytest.approx(np.eye(M), abs=1e-8)


def test_rng170_stays_symmetric():
    rng = np.random.default_rng(27)
    M = 3
    Pinv = np.eye(M) * 5.0
    for _ in range(20):
        Pinv = inv_rec(Pinv, rng.standard_normal(M), 0.9)["array"]
    assert np.allclose(Pinv, Pinv.T)


def test_rng170_zero_reference_is_pure_rescaling():
    """r = 0 kills the correction term, leaving Phi^-1(n) = lam^-1 Phi^-1(n-1)."""
    Pinv = np.array([[2.0, 1.0], [1.0, 3.0]])
    got = inv_rec(Pinv, np.zeros(2), 0.5)["array"]
    assert got == pytest.approx(2.0 * Pinv)


def test_rng170_rejects_bad_lambda_and_shape():
    with pytest.raises(ValueError, match="0 < lam <= 1"):
        inv_rec(np.eye(2), np.ones(2), 0.0)
    with pytest.raises(ValueError, match="must have length"):
        inv_rec(np.eye(3), np.ones(2), 0.9)
