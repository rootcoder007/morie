"""Tests for grxeng.geron_softmax_cost_gradient."""

from morie.fn import _array_core as np

import math

from morie.fn.grxeng import geron_softmax_cost_gradient


def test_grxeng_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    m, n, K = 40, 3, 3
    X = rng.normal(0, 1, (m, n))
    Y = rng.integers(0, K, size=m)
    theta = rng.normal(0, 1, (n, K))
    result = geron_softmax_cost_gradient(X, Y, theta)
    # RichResult behaves like a dict
    assert isinstance(result, dict)
    # Verify payload keys
    for key in ("gradient", "probabilities", "gradient_norm", "estimate", "n", "method"):
        assert key in result
    # Check shapes
    grad = np.array(result["gradient"])
    probs = np.array(result["probabilities"])
    assert grad.shape == (n, K)
    assert probs.shape == (m, K)
    # Gradient norm should be finite
    assert math.isfinite(result["gradient_norm"])
    # n should match number of instances
    assert result["n"] == m
    # method should be a string
    assert isinstance(result["method"], str)
    # Probabilities should sum to 1 per instance
    row_sums = np.sum(probs, axis=1).tolist()
    assert all(abs(s - 1.0) < 1e-6 for s in row_sums)
    # Gradient rows (summed across classes) should be zero for each feature
    feature_sums = np.sum(grad, axis=1).tolist()
    assert all(abs(s) < 1e-9 for s in feature_sums)


def test_grxeng_edge():
    """Test edge cases: one‑hot encoded labels."""
    rng = np.random.default_rng(1)
    m, n, K = 20, 3, 3
    X = rng.normal(0, 1, (m, n))
    labels = rng.integers(0, K, size=m)
    Y_onehot = np.eye(K)[labels]  # one‑hot encoding, shape (m, K)
    theta = rng.normal(0, 1, (n, K))
    result = geron_softmax_cost_gradient(X, Y_onehot, theta)
    # Basic key checks
    assert "gradient" in result
    assert "probabilities" in result
    # Shape checks
    grad = np.array(result["gradient"])
    probs = np.array(result["probabilities"])
    assert grad.shape == (n, K)
    assert probs.shape == (m, K)
    # Finite gradient norm
    assert math.isfinite(result["gradient_norm"])
    # Number of instances matches
    assert result["n"] == m
    # Probabilities sum to 1 per instance
    row_sums = np.sum(probs, axis=1).tolist()
    assert all(abs(s - 1.0) < 1e-6 for s in row_sums)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    r = geron_softmax_cost_gradient([[1.0]], [0], [[0.0, 0.0, 0.0]])
    grad_row = r["gradient"][0]
    assert [round(v, 6) for v in grad_row] == [-0.666667, 0.333333, 0.333333]
    assert abs(sum(grad_row)) < 1e-12
