"""Tests for mtlmm.multi_trait_lmm."""

from morie.fn import _array_core as np

from morie.fn.mtlmm import multi_trait_lmm


def test_mtlmm_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_z = np.random.default_rng(41)

    n, t, p, m = 40, 3, 3, 10
    Y = rng_y.normal(0, 1, (n, t))
    X = rng_x.normal(0, 1, (n, p))
    Z = rng_z.normal(0, 1, (n, m))
    A = np.eye(m)
    R_T = np.eye(t)

    result = multi_trait_lmm(Y, X, Z, A, R_T)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_mtlmm_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_z = np.random.default_rng(41)

    n, t, p, m = 40, 2, 3, 10
    Y = rng_y.normal(0, 1, (n, t))
    X = rng_x.normal(0, 1, (n, p))
    Z = rng_z.normal(0, 1, (n, m))
    A = np.eye(m)
    R_T = np.eye(t)

    result = multi_trait_lmm(Y, X, Z, A, R_T)
    assert isinstance(result, dict)
