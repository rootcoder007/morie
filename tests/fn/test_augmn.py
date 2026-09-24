"""Tests for augmn.albert_chib_augmentation."""

from morie.fn import _array_core as np

from morie.fn.augmn import albert_chib_augmentation


def test_augmn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    q = 5
    # y_bin must contain only 0 or 1
    y_bin = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    Z = rng.normal(0, 1, (n, q))
    result = albert_chib_augmentation(y_bin, X, Z)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "z_samples" in result
    assert "beta_samples" in result
    assert "b" in result
    assert len(result["z_samples"]) == n
    assert len(result["beta_samples"]) == p
    assert len(result["b"]) == q


def test_augmn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    y_bin = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, p))
    # Z=None -> identity matrix, so b has length n
    result = albert_chib_augmentation(y_bin, X)
    assert isinstance(result, dict)
    assert "b" in result
    assert len(result["b"]) == n
    assert len(result["z_samples"]) == n
    assert len(result["beta_samples"]) == p
