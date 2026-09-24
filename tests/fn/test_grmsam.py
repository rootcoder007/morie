"""Tests for grmsam.graded_response_samejima."""

import math

from morie.fn import _array_core as np

from morie.fn.grmsam import graded_response_samejima


def test_grmsam_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 40
    m = 3  # 4 categories (0..3)

    # Observed categories, 0-based (0 .. m)
    y = list(rng.integers(0, m + 1, n))

    # Person abilities, same length as y
    theta = list(rng.normal(0, 1, n))

    # Item slope (a single float)
    a = 1.5

    # m strictly increasing category thresholds
    b_k = [-1.0, 0.0, 1.0]

    result = graded_response_samejima(y, theta, a, b_k)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["categories"] == m + 1
    assert math.isfinite(result["estimate"])
    assert 0 < result["estimate"] < 1
    assert math.isfinite(result["loglik"])
    assert len(result["p_observed"]) == n
    assert len(result["probs_first"]) > 0
    assert isinstance(result["method"], str) and len(result["method"]) > 0


def test_grmsam_edge():
    """Test edge case with minimal valid input."""
    # Smallest valid configuration: one person, two categories (m=1)
    y = [0]
    theta = [0.0]
    a = 1.0
    b_k = [0.5]

    result = graded_response_samejima(y, theta, a, b_k)
    assert isinstance(result, dict)
    assert result["n"] == 1
    assert result["categories"] == 2
    assert math.isfinite(result["loglik"])
    assert 0 < result["estimate"] < 1
    assert len(result["p_observed"]) == 1
    assert len(result["probs_first"]) > 0
