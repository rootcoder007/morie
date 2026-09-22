"""Tests for baynav.variational_nf."""

import numpy as real_np

from morie.fn import _array_core as np

from morie.fn.baynav import variational_nf


def _planar_layer(dim, rng):
    """Build one (u, w, b) layer for a planar flow of given dim."""
    u = rng.normal(0.0, 1.0, dim)
    w = rng.normal(0.0, 1.0, dim)
    b = rng.normal(0.0, 0.1)  # scalar bias
    return (u, w, b)


def test_baynav_basic():
    """Test basic functionality of variational_nf on a single 2D sample."""
    rng = np.random.default_rng(42)
    dim = 2

    # z0: a single sample vector (1D, length == dim)
    z0 = rng.normal(0.0, 1.0, dim)

    # log_q0: a scalar log-density value at z0 under the base distribution
    log_q0 = float(rng.normal(0.0, 1.0))

    # layers: an iterable of (u, w, b) triples for planar flows
    num_layers = 3
    layers = [_planar_layer(dim, rng) for _ in range(num_layers)]

    result = variational_nf(z0, log_q0, layers)

    assert isinstance(result, dict)

    # The function must return the documented keys
    for key in ("estimate", "log_q", "z", "log_dets", "depth",
                "method", "note"):
        assert key in result, f"missing documented key: {key}"

    # Depth matches the number of layers supplied
    assert result["depth"] == num_layers

    # log_q and estimate must agree and be scalars
    assert result["estimate"] == result["log_q"]

    # Independent recomputation of the change-of-variables formula:
    # log_q_K = log_q_0 - sum_k log|det J_k|
    expected_log_q = log_q0 - sum(result["log_dets"])
    assert real_np.isclose(float(result["log_q"]), expected_log_q)

    # log_dets has one entry per layer
    assert len(result["log_dets"]) == num_layers


def test_baynav_edge():
    """Test edge cases: zero layers leaves density untouched (mean-field)."""
    rng = np.random.default_rng(7)
    dim = 3

    z0 = rng.normal(0.0, 1.0, dim)
    log_q0 = float(rng.normal(0.0, 1.0))

    # No layers -> the flow is the identity, density is unchanged
    result = variational_nf(z0, log_q0, [])

    assert isinstance(result, dict)
    assert result["depth"] == 0
    assert result["log_dets"] == []
    assert real_np.isclose(float(result["log_q"]), log_q0)
    assert real_np.isclose(float(result["estimate"]), log_q0)
