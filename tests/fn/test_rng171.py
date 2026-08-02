"""rng171: RLS Kalman-like gain (Rangayyan 2024, Eq. 3.217, p. 188)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng171 import rangayyan_ch3_rls_kalman_gain as gain
from morie.fn.rng172 import rangayyan_ch3_rls_p_recursion as p_rec
from morie.fn.rng173 import rangayyan_ch3_rls_gain_identity as gain_id


def test_rng171_satisfies_the_gain_identity_of_eq_3221():
    """k(n) from P(n-1) via Eq. (3.217) must equal P(n) r(n) via Eq. (3.221).

    The book derives 3.221 independently of 3.217, so this ties the two
    modules together through the P recursion rather than restating either.
    """
    rng = np.random.default_rng(43)
    M, lam = 4, 0.9
    P = np.eye(M) * 8.0
    for _ in range(20):
        r = rng.standard_normal(M)
        k = gain(P, r, lam)["array"]
        P = p_rec(P, k, r, lam)["array"]
        assert gain_id(P, r)["array"] == pytest.approx(k, rel=1e-9)


def test_rng171_single_step_by_hand():
    """P = I, r = (1,0), lam = 0.5 -> k = 2*(1,0)/(1 + 2) = (2/3, 0)."""
    got = gain(np.eye(2), np.array([1.0, 0.0]), 0.5)["array"]
    assert got == pytest.approx([2.0 / 3.0, 0.0])


def test_rng171_zero_reference_gives_zero_gain():
    """No new information means no correction."""
    got = gain(np.eye(3) * 5.0, np.zeros(3), 0.9)["array"]
    assert got == pytest.approx(np.zeros(3))


def test_rng171_is_not_the_mean_of_P():
    """Regression guard: the previous body returned float(np.mean(P)).

    That stub was green in the suite because its test asserted only that the
    key "estimate" existed.
    """
    r = gain(np.eye(3) * 4.0, np.array([1.0, 2.0, 3.0]), 0.8)
    assert "array" in r
    assert np.asarray(r["array"]).shape == (3,)


def test_rng171_rejects_bad_lambda_and_shape():
    with pytest.raises(ValueError, match="0 < lam <= 1"):
        gain(np.eye(2), np.ones(2), 0.0)
    with pytest.raises(ValueError, match="must have length"):
        gain(np.eye(3), np.ones(2), 0.9)
