"""Tests for mmdsf."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mmdsf import metric_mds_torgerson


def _planted(seed=0, n=10, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    diff = X[:, None, :] - X[None, :, :]
    return X, np.sqrt((diff**2).sum(axis=2))


def test_mmdsf_basic():
    X, D = _planted()
    out = metric_mds_torgerson(D, n_dims=2)
    diff = out["coordinates"][:, None, :] - out["coordinates"][None, :, :]
    assert np.sqrt((diff**2).sum(axis=2)) == pytest.approx(D, abs=1e-8)
    assert out["explained"] == pytest.approx(1.0, abs=1e-8)


def test_mmdsf_edge():
    _, D = _planted()
    with pytest.raises(ValueError):
        metric_mds_torgerson(D, n_dims=0)
    with pytest.raises(ValueError):
        metric_mds_torgerson(D + np.eye(D.shape[0]))  # nonzero diagonal
