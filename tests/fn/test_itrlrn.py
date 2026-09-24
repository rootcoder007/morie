"""Tests for itrlrn.iterative_q_learning."""

import math

from morie.fn import _array_core as np

from morie.fn.itrlrn import iterative_q_learning


def _make_panel(seed, n_subjects=40, T_stages=5, k=3):
    rng = np.random.default_rng(seed)
    n = n_subjects * T_stages
    state = rng.normal(0, 1, (n, k))
    action = [float(x) for x in rng.integers(0, 2, n)]
    reward = rng.normal(0, 1, n)
    time = []
    for t in range(T_stages):
        time.extend([float(t)] * n_subjects)
    return state, action, reward, time


def test_itrlrn_basic():
    """Test basic functionality."""
    state, action, reward, time = _make_panel(42)
    result = iterative_q_learning(state, action, reward, time)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value" in result
    assert "stage_value" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["value"])
    assert len(result["stage_value"]) == 5


def test_itrlrn_edge():
    """Test edge cases."""
    state, action, reward, time = _make_panel(123)
    result = iterative_q_learning(state, action, reward, time, gamma=0.5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
