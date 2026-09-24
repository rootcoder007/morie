"""Tests for icc1k.icc_one_way_average."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.icc1k import icc_one_way_average


def test_icc1k_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n_targets = 10
    k = 4
    # Build cluster: each of n_targets targets has k ratings
    cluster = []
    for i in range(n_targets):
        cluster.extend([i] * k)
    # Generate ratings with a between-target mean effect
    target_means = rng.normal(0, 1, n_targets)
    y = [target_means[i] + r for i in range(n_targets) for r in rng.normal(0, 1, k)]
    result = icc_one_way_average(y, cluster)
    # Result should be dict-like with the documented keys
    assert isinstance(result, dict)
    expected_keys = ("value", "icc_single", "k", "n", "MSR", "MSW",
                     "case", "design_assumption", "smallest_because", "method")
    for key in expected_keys:
        assert key in result
    # The ICC estimates must be finite numbers
    assert math.isfinite(result["value"])
    assert math.isfinite(result["icc_single"])
    # The recovered design parameters match the inputs
    assert result["k"] == k
    assert result["n"] == n_targets


def test_icc1k_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    # Unbalanced cluster: target 0 has 2 ratings, target 1 has 3, etc.
    cluster = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    y = rng.normal(0, 1, 10)
    with pytest.raises(ValueError):
        icc_one_way_average(y, cluster)
