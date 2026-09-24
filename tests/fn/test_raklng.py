"""Tests for raklng.raking_ratio."""

import math

from morie.fn import _array_core as np

from morie.fn.raklng import raking_ratio


def test_raklng_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 100
    y = [float(v) for v in rng.normal(0, 1, n)]
    w_raw = [float(v) for v in rng.normal(0, 1, n)]
    weights = [abs(v) + 0.1 for v in w_raw]

    # Two consistent margins (both sum to 100)
    margin1_labels = ['A' if i < 50 else 'B' for i in range(n)]
    margin1_targets = {'A': 30.0, 'B': 70.0}

    margin2_labels = ['X' if i < 60 else 'Y' for i in range(n)]
    margin2_targets = {'X': 55.0, 'Y': 45.0}

    margins = [(margin1_labels, margin1_targets), (margin2_labels, margin2_targets)]

    result = raking_ratio(y, weights, margins)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "weights" in result
    assert "iterations" in result
    assert "max_margin_error" in result
    assert "N" in result
    assert "n" in result
    assert math.isfinite(result["estimate"])


def test_raklng_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 4
    y = [float(v) for v in rng.normal(0, 1, n)]
    weights = [0.1, 0.2, 0.3, 0.4]

    # Single margin with minimal data
    labels = ['A', 'A', 'B', 'B']
    targets = {'A': 10.0, 'B': 20.0}

    margins = [(labels, targets)]

    result = raking_ratio(y, weights, margins)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
