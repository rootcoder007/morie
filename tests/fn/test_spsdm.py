"""Tests for spsdm.schabenberger_spatial_durbin_model.

spsdm delegates to sgdbn.spatial_durbin_model; the contract these tests pin is that
the delegation reaches the same estimator with the arguments mapped
correctly, so the two must agree exactly.
"""

import numpy as np

from morie.fn.spsdm import schabenberger_spatial_durbin_model
from morie.fn.sgdbn import spatial_durbin_model


def _fixture():
    rng = np.random.default_rng(11)
    n = 40
    W = (rng.random((n, n)) < 0.2).astype(float)
    np.fill_diagonal(W, 0.0)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    X = rng.random((n, 2))
    y = 0.4 * (W @ X[:, 0]) + X @ np.array([1.0, -1.0]) + rng.normal(0, 0.3, n)
    return X, y, W


def test_spsdm_matches_sgdbn():
    """The delegation must reproduce the implemented estimator exactly."""
    X, y, W = _fixture()
    got = schabenberger_spatial_durbin_model(X, y, W)
    ref = spatial_durbin_model(y, X, W)
    assert str(got) == str(ref)


def test_spsdm_edge():
    """Degenerate input is handled by the implemented estimator, not here."""
    X, y, W = _fixture()
    # argument ORDER differs between the two front ends; getting it wrong
    # would silently fit the transpose problem
    got = schabenberger_spatial_durbin_model(X, y, W)
    assert got.statistic == spatial_durbin_model(y, X, W).statistic
    assert -1.0 < got.statistic < 1.0
