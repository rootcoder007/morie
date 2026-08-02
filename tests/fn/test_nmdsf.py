"""Tests for nmdsf."""

from morie.fn import _array_core as np
import pytest

from morie.fn.nmdsf import nonmetric_mds


def _planted(seed=0, n=10, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    diff = X[:, None, :] - X[None, :, :]
    return X, np.sqrt((diff**2).sum(axis=2))


def test_nmdsf_basic():
    _, D = _planted(2, n=12)
    out = nonmetric_mds(D**3, n_dims=2, max_iter=200)  # monotone distortion
    assert out["stress"] < 0.05


def test_nmdsf_edge():
    _, D = _planted(2)
    with pytest.raises(ValueError):
        nonmetric_mds(D, n_dims=0)
    with pytest.raises(ValueError):
        nonmetric_mds(D[:4, :5])
