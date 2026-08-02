"""rng168: RLS cross-correlation recursion (Rangayyan 2024, Eq. 3.212, p. 187)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng168 import rangayyan_ch3_rls_theta_recursion as theta_rec


def test_rng168_recursion_reproduces_the_closed_form_sum():
    """Iterating Eq. (3.212) must equal the definition Eq. (3.209),
    Theta(n) = sum_{i=1..n} lam^(n-i) r(i) x(i)."""
    rng = np.random.default_rng(14)
    M, n_steps, lam = 3, 10, 0.85
    R = rng.standard_normal((n_steps, M))
    xs = rng.standard_normal(n_steps)
    Theta = np.zeros(M)
    for i in range(n_steps):
        Theta = theta_rec(Theta, R[i], xs[i], lam)["array"]
    closed = sum(lam ** (n_steps - 1 - i) * R[i] * xs[i] for i in range(n_steps))
    assert Theta == pytest.approx(closed, rel=1e-12)


def test_rng168_single_step_by_hand():
    Theta = np.array([1.0, 2.0])
    r = np.array([3.0, -1.0])
    got = theta_rec(Theta, r, 2.0, 0.5)["array"]
    assert got == pytest.approx([0.5 * 1.0 + 6.0, 0.5 * 2.0 - 2.0])


def test_rng168_zero_primary_input_only_forgets():
    """x(n) = 0 contributes nothing, leaving pure geometric decay."""
    Theta = np.array([4.0, -8.0])
    got = theta_rec(Theta, np.ones(2), 0.0, 0.75)["array"]
    assert got == pytest.approx(0.75 * Theta)


def test_rng168_is_linear_in_the_primary_input():
    Theta = np.zeros(3)
    r = np.array([1.0, 2.0, 3.0])
    a = theta_rec(Theta, r, 1.0, 0.9)["array"]
    b = theta_rec(Theta, r, 2.0, 0.9)["array"]
    assert b == pytest.approx(2.0 * a)


def test_rng168_rejects_a_signal_for_the_scalar_primary_input():
    """Eq. (3.212) updates ONE time step; x(n) is a sample, not a waveform."""
    with pytest.raises(ValueError, match="scalar sample"):
        theta_rec(np.zeros(2), np.ones(2), np.arange(5.0), 0.9)


def test_rng168_rejects_bad_lambda_and_shape():
    with pytest.raises(ValueError, match="0 < lam <= 1"):
        theta_rec(np.zeros(2), np.ones(2), 1.0, 1.2)
    with pytest.raises(ValueError, match="same length as Theta"):
        theta_rec(np.zeros(3), np.ones(2), 1.0, 0.9)
