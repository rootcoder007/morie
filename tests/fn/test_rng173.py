"""rng173: RLS gain identity k(n) = P(n) r(n) (Rangayyan 2024, Eq. 3.221, p. 188)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng173 import rangayyan_ch3_rls_gain_identity as gain_id


def test_rng173_is_a_plain_matrix_vector_product():
    P = np.array([[2.0, 1.0], [0.0, 3.0]])
    r = np.array([1.0, -1.0])
    assert gain_id(P, r)["array"] == pytest.approx([1.0, -3.0])


def test_rng173_identity_matrix_returns_the_reference():
    r = np.array([1.5, -2.5, 0.0])
    assert gain_id(np.eye(3), r)["array"] == pytest.approx(r)


def test_rng173_is_linear_in_r():
    rng = np.random.default_rng(53)
    P = rng.standard_normal((4, 4))
    a, b = rng.standard_normal(4), rng.standard_normal(4)
    lhs = gain_id(P, a + b)["array"]
    rhs = gain_id(P, a)["array"] + gain_id(P, b)["array"]
    assert lhs == pytest.approx(rhs)


def test_rng173_is_not_the_mean_of_P():
    """Regression guard: previous body was float(np.mean(P)) under "estimate"."""
    out = gain_id(np.eye(3) * 4.0, np.array([1.0, 2.0, 3.0]))
    assert np.asarray(out["array"]).shape == (3,)


def test_rng173_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must have length"):
        gain_id(np.eye(3), np.ones(2))
    with pytest.raises(ValueError, match="square matrix"):
        gain_id(np.ones((2, 3)), np.ones(3))
