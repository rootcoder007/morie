"""Tests for rginf.rangayyan_infomax_ica."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_infomax_ica


def test_rginf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 5))
    n_components = 3
    lr = 0.1
    max_iter = 100
    result = rangayyan_infomax_ica(X, n_components, lr, max_iter)
    assert isinstance(result, dict)
    assert "components" in result or "sources" in result or "estimate" in result


def test_rginf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 5))
    n_components = 3
    lr = 0.01
    max_iter = 50
    result = rangayyan_infomax_ica(X, n_components, lr, max_iter)
    assert isinstance(result, dict)
