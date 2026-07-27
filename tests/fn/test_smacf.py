"""Tests for smacf."""

import numpy as np
import pytest

from morie.fn.smacf import smacof_algorithm


def _planted(seed=0, n=10, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    diff = X[:, None, :] - X[None, :, :]
    return X, np.sqrt((diff**2).sum(axis=2))


def test_smacf_basic():
    _, D = _planted(1)
    out = smacof_algorithm(D, n_dims=2, max_iter=200)
    assert np.all(np.diff(out["stress_path"]) <= 1e-9)  # majorization
    assert out["stress"] < 1e-6


def test_smacf_edge():
    _, D = _planted(1)
    with pytest.raises(ValueError):
        smacof_algorithm(D, n_dims=0)
    with pytest.raises(ValueError):
        smacof_algorithm(D[:4, :5])
