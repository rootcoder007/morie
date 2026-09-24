"""Tests for opttre.optimal_tree_regime."""

import math

from morie.fn import _array_core as np

from morie.fn.opttre import optimal_tree_regime


def test_opttre_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    n = 40
    p = 3
    y = rng.normal(0, 1, n)
    # Ensure both treatments are present
    A = [0.0, 1.0] * (n // 2) + [0.0] * (n % 2)
    W = rng.normal(0, 1, (n, p))
    result = optimal_tree_regime(y, A, W)
    assert isinstance(result, dict)
    expected_keys = {
        "estimate",
        "value",
        "value_all_treated",
        "value_all_control",
        "rule",
        "split_var",
        "split_point",
        "n_leaves",
        "depth",
        "n",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n"] == n
    assert len(result["rule"]) == n
    assert math.isfinite(result["estimate"])


def test_opttre_edge():
    """Test edge cases."""
    rng = np.random.default_rng(1)
    n = 5
    p = 2
    y = rng.normal(0, 1, n)
    # Both treatments present
    A = [0.0, 1.0, 0.0, 1.0, 0.0]
    W = rng.normal(0, 1, (n, p))
    result = optimal_tree_regime(y, A, W)
    assert isinstance(result, dict)
    expected_keys = {
        "estimate",
        "value",
        "value_all_treated",
        "value_all_control",
        "rule",
        "split_var",
        "split_point",
        "n_leaves",
        "depth",
        "n",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n"] == n
    assert len(result["rule"]) == n
