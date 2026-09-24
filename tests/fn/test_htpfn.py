"""Tests for htpfn.htp_functional_predictor."""

from morie.fn import _array_core as np

from morie.fn.htpfn import htp_functional_predictor


def test_htpfn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 40
    p = 3
    m = 10
    y = rng.normal(0, 1, n)
    markers = rng.integers(0, 3, (n, p))
    W_functional = rng.normal(0, 1, (n, m))
    result = htp_functional_predictor(y, markers, W_functional, n_basis=4)
    assert isinstance(result, dict)
    assert "beta_func" in result
    assert "g_hat" in result
    assert len(result["beta_func"]) == m
    assert len(result["g_hat"]) == n


def test_htpfn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 40
    p = 3
    m = 6
    y = rng.normal(0, 1, n)
    markers = rng.integers(0, 3, (n, p))
    W_functional = rng.normal(0, 1, (n, m))
    result = htp_functional_predictor(y, markers, W_functional, n_basis=2, lam=0.5)
    assert isinstance(result, dict)
    assert "beta_func" in result
    assert "g_hat" in result
    assert len(result["beta_func"]) == m
    assert len(result["g_hat"]) == n
