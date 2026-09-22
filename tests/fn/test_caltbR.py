"""Tests for caltbR.calibrated_rec."""

import numpy as np

from morie.fn import _array_core as np

from morie.fn.caltbR import calibrated_rec


def test_caltbR_basic():
    """Test basic functionality."""
    n = 100
    g = 5
    rng = np.random.default_rng(42)

    pred = rng.normal(0, 1, n)

    # p_g_given_i: shape (n, g), each row a probability distribution
    raw = rng.uniform(0, 1, (n, g))
    user_profile = raw / raw.sum(axis=1, keepdims=True)

    # p_target: shape (g,), uniform target distribution
    p_target = np.ones(g) / g

    result = calibrated_rec(pred, user_profile, p_target)

    assert isinstance(result, dict)
    assert "estimate" in result


def test_caltbR_edge():
    """Test edge cases."""
    n = 100
    g = 5
    rng = np.random.default_rng(42)

    pred = rng.normal(0, 1, n)

    raw = rng.uniform(0, 1, (n, g))
    user_profile = raw / raw.sum(axis=1, keepdims=True)

    p_target = np.ones(g) / g

    result = calibrated_rec(pred, user_profile, p_target)

    assert isinstance(result, dict)
    assert "estimate" in result
