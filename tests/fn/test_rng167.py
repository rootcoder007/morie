"""rng167: RLS autocorrelation recursion (Rangayyan 2024, Eq. 3.211, p. 187)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsaadapt import rangayyan_ch3_rls_phi_recursion as phi_rec


def test_rng167_recursion_reproduces_the_closed_form_sum():
    """Iterating Eq. (3.211) must equal the definition Eq. (3.208).

    Phi(n) = sum_{i=1..n} lam^(n-i) r(i) r'(i). This is the check the
    recursion exists to satisfy, and it fails for any wrong power of lambda.
    """
    rng = np.random.default_rng(13)
    M, n_steps, lam = 3, 12, 0.9
    R = rng.standard_normal((n_steps, M))
    Phi = np.zeros((M, M))
    for i in range(n_steps):
        Phi = phi_rec(Phi, R[i], lam)["array"]
    closed = sum(lam ** (n_steps - 1 - i) * np.outer(R[i], R[i]) for i in range(n_steps))
    assert Phi == pytest.approx(closed, rel=1e-12)


def test_rng167_single_step_by_hand():
    Phi = np.array([[1.0, 2.0], [2.0, 5.0]])
    r = np.array([1.0, -1.0])
    got = phi_rec(Phi, r, 0.5)["array"]
    assert got == pytest.approx(0.5 * Phi + np.array([[1.0, -1.0], [-1.0, 1.0]]))


def test_rng167_stays_symmetric_and_psd():
    """An outer product is rank-1 PSD and lam > 0, so the sum stays PSD."""
    rng = np.random.default_rng(6)
    M, lam = 4, 0.95
    Phi = np.eye(M) * 1e-3
    for _ in range(40):
        Phi = phi_rec(Phi, rng.standard_normal(M), lam)["array"]
    assert np.allclose(Phi, Phi.T)
    assert np.all(np.linalg.eigvalsh(Phi) > -1e-12)


def test_rng167_lambda_one_is_a_plain_running_sum():
    """lam = 1 means infinite memory: no forgetting at all."""
    rng = np.random.default_rng(10)
    M = 2
    R = rng.standard_normal((5, M))
    Phi = np.zeros((M, M))
    for i in range(5):
        Phi = phi_rec(Phi, R[i], 1.0)["array"]
    assert Phi == pytest.approx(sum(np.outer(v, v) for v in R))


def test_rng167_rejects_lambda_outside_the_book_range():
    """Rangayyan p. 186 bounds the forgetting factor as 0 < lam <= 1."""
    with pytest.raises(ValueError, match="0 < lam <= 1"):
        phi_rec(np.eye(2), np.ones(2), 1.5)
    with pytest.raises(ValueError, match="0 < lam <= 1"):
        phi_rec(np.eye(2), np.ones(2), 0.0)


def test_rng167_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must have length"):
        phi_rec(np.eye(3), np.ones(2), 0.9)
