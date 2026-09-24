"""Tests for hmdbd.geron_decision_boundary."""

import math

from morie.fn import _array_core as np
from morie.fn.hmdbd import geron_decision_boundary


def test_hmdbd_basic():
    """Test basic functionality with random data."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X_grid = rng.normal(0, 1, (n, p))
    # theta[0] is bias, theta[1:] are weights -> size p+1 with fit_intercept
    theta = [0.5, -1.0, 0.7, 0.2]
    result = geron_decision_boundary(theta, X_grid)
    assert isinstance(result, dict)
    # Check keys mentioned in the docstring return description
    for key in ("scores", "signed_distance", "labels", "probabilities",
                "on_boundary", "normal", "line"):
        assert key in result
    # Shape consistency: one entry per grid point
    assert len(result["scores"]) == n
    assert len(result["signed_distance"]) == n
    assert len(result["labels"]) == n
    assert len(result["probabilities"]) == n
    assert len(result["on_boundary"]) == n
    # Probabilities must lie in [0, 1]
    for prob in result["probabilities"]:
        assert 0.0 <= prob <= 1.0
    # Labels are 0 or 1
    for label in result["labels"]:
        assert label in (0, 1)
    # Scores and signed distances are finite
    for s in result["scores"]:
        assert math.isfinite(s)
    for d in result["signed_distance"]:
        assert math.isfinite(d)


def test_hmdbd_edge():
    """Test edge case: a point landing exactly on the decision boundary."""
    # From docstring: boundary x1 + x2 = 1 corresponds to theta = [-1, 1, 1]
    theta = [-1.0, 1.0, 1.0]
    X_grid = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
    result = geron_decision_boundary(theta, X_grid)
    assert isinstance(result, dict)
    # Labels: origin below, (1,1) above, midpoint on the line
    assert list(result["labels"]) == [0, 1, 0]
    # Only the midpoint is flagged as on the boundary
    assert list(result["on_boundary"]) == [False, False, True]
    # Probability at the boundary is exactly one half
    assert abs(result["probabilities"][2] - 0.5) < 1e-12
    # Signed distance of the midpoint is zero
    assert abs(result["signed_distance"][2]) < 1e-9
    # The 2-D line is x2 = -x1 + 1: slope -1, intercept 1
    assert abs(result["line"][0] - (-1.0)) < 1e-6
    assert abs(result["line"][1] - 1.0) < 1e-6
