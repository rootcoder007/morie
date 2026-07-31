"""Tests for spsar.schabenberger_sar_model.

The generated version of this file passed `w` as a 1-D array of 100
draws rather than an (n, n) weights matrix, so it raised "shape mismatch
among x, y, w" and had never passed.
"""

import numpy as np
import pytest

from morie.fn.spsar import schabenberger_sar_model
from morie.fn.sarla import spatial_ar_lag
from morie.fn._schab_rho import rho_bounds


def _case(n=24):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.random(n)])
    y = np.linalg.solve(np.eye(n) - 0.4 * W, X @ np.array([1.0, -0.5])
                        + rng.normal(0, 0.3, n))
    return X, y, W


def test_spsar_delegates_to_spatial_ar_lag():
    X, y, W = _case()
    assert float(schabenberger_sar_model(X, y, W)["rho"]) == \
        float(spatial_ar_lag(X, y, W)["rho"])


def test_spsar_rho_lies_inside_the_valid_parameter_space():
    """|I - rho W| must stay positive; the bound is 1/theta (eq 6.48)."""
    X, y, W = _case()
    lo, hi = rho_bounds(W, "identity")
    rho = float(schabenberger_sar_model(X, y, W)["rho"])
    assert lo < rho < hi


def test_spsar_recovers_a_known_rho():
    n = 24
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(n), rng.random(n)])
    est = []
    for _ in range(40):
        y = np.linalg.solve(np.eye(n) - 0.5 * W,
                            X @ np.array([1.0, -0.5]) + rng.normal(0, 0.3, n))
        est.append(float(schabenberger_sar_model(X, y, W)["rho"]))
    assert np.mean(est) == pytest.approx(0.5, abs=0.2)


def test_spsar_rejects_mismatched_shapes():
    X, y, W = _case()
    with pytest.raises(ValueError):
        schabenberger_sar_model(X, y, W[:-1, :-1])
