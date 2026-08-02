"""rng172: RLS P(n) recursion via the gain vector (Rangayyan 2024, Eq. 3.218, p. 188)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng167 import rangayyan_ch3_rls_phi_recursion as phi_rec
from morie.fn.rng170 import rangayyan_ch3_rls_inverse_recursion as inv_rec
from morie.fn.rng172 import rangayyan_ch3_rls_p_recursion as p_rec


def _gain(P, r, lam):
    """Eq. (3.217): k(n) = lam^-1 P(n-1) r(n) / (1 + lam^-1 r'(n) P(n-1) r(n)).

    Deliberately local, NOT imported from morie.fn.rng171. That module is named
    for this equation and prints it correctly in its docstring, but its body is
    ``float(np.mean(P))`` -- the shared mean-and-standard-error stub. It is
    green in the suite only because its test asserts ``"estimate" in result``,
    which the stub happens to provide, so it sits in the false-negative
    quadrant the audit README puts out of scope for the red series. Importing
    it here would make these tests assert against a mean.
    """
    inv_lam = 1.0 / lam
    num = inv_lam * (P @ r)
    return num / (1.0 + inv_lam * float(r @ P @ r))


def test_rng172_equals_the_riccati_form_of_eq_3215():
    """Eq. (3.218) is Eq. (3.215) rewritten; they must agree step for step."""
    rng = np.random.default_rng(29)
    M, lam = 4, 0.9
    P = np.eye(M) * 10.0
    P_ref = P.copy()
    for _ in range(25):
        r = rng.standard_normal(M)
        P = p_rec(P, _gain(P, r, lam), r, lam)["array"]
        P_ref = inv_rec(P_ref, r, lam)["array"]
        assert P == pytest.approx(P_ref, rel=1e-9)


def test_rng172_satisfies_eq_3221_k_equals_P_r():
    """Eq. (3.221): k(n) = P(n) r(n).

    The book derives this by comparing Eq. (3.220) with Eq. (3.218), so it is
    an independent consequence of the update -- exactly the identity that
    catches a k inconsistent with P, r and lambda.
    """
    rng = np.random.default_rng(37)
    M, lam = 3, 0.85
    P = np.eye(M) * 4.0
    for _ in range(15):
        r = rng.standard_normal(M)
        k = _gain(P, r, lam)
        P = p_rec(P, k, r, lam)["array"]
        assert P @ r == pytest.approx(k, rel=1e-9)


def test_rng172_tracks_the_inverse_of_the_Phi_recursion():
    """P(n) is Phi^-1(n) by Eq. (3.216); check against the uninverted chain."""
    rng = np.random.default_rng(41)
    M, lam = 3, 0.95
    delta_inv = 100.0
    P = np.eye(M) * delta_inv
    Phi = np.eye(M) / delta_inv
    for _ in range(30):
        r = rng.standard_normal(M)
        P = p_rec(P, _gain(P, r, lam), r, lam)["array"]
        Phi = phi_rec(Phi, r, lam)["array"]
    assert Phi @ P == pytest.approx(np.eye(M), abs=1e-7)


def test_rng172_zero_gain_is_pure_rescaling():
    P = np.array([[3.0, 1.0], [1.0, 2.0]])
    got = p_rec(P, np.zeros(2), np.ones(2), 0.5)["array"]
    assert got == pytest.approx(2.0 * P)


def test_rng172_rejects_bad_lambda_and_shapes():
    with pytest.raises(ValueError, match="0 < lam <= 1"):
        p_rec(np.eye(2), np.ones(2), np.ones(2), 2.0)
    with pytest.raises(ValueError, match="k must have length"):
        p_rec(np.eye(3), np.ones(2), np.ones(3), 0.9)
    with pytest.raises(ValueError, match="r must have length"):
        p_rec(np.eye(3), np.ones(3), np.ones(2), 0.9)
