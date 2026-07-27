"""Tests for shrpd."""

import numpy as np
import pytest

from morie.fn.shrpd import shepard_diagram


def _planted(seed=0, n=10, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    diff = X[:, None, :] - X[None, :, :]
    return X, np.sqrt((diff**2).sum(axis=2))


def test_shrpd_basic():
    _, D = _planted(3, n=8)
    out = shepard_diagram(D**2, D)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert np.all(np.diff(out["monotone_fit"]) >= -1e-12)


def test_shrpd_edge():
    _, D = _planted(3)
    with pytest.raises(ValueError):
        shepard_diagram(D[:4, :5], D[:4, :5])
