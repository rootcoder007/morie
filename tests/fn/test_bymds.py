"""Tests for bymds."""

import numpy as np
import pytest

from morie.fn.bymds import bayesian_mds


def test_bymds_basic():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(7, 2))
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    out = bayesian_mds(D, n_dims=2, n_iter=300, burnin=100, seed=0, step=0.03)
    est = out["coordinates"]
    de = np.sqrt(((est[:, None, :] - est[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(7, 1)
    assert np.corrcoef(de[iu], D[iu])[0, 1] > 0.95


def test_bymds_edge():
    D = np.abs(np.subtract.outer(np.arange(5.0), np.arange(5.0)))
    with pytest.raises(ValueError):
        bayesian_mds(D, n_iter=50, burnin=100)
    with pytest.raises(ValueError):
        bayesian_mds(D[:3, :4])
