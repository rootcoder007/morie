"""Tests for gh_c3_2.ghosal_stochastic_proc_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_2 import ghosal_stochastic_proc_prior


def test_gh_c3_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_stochastic_proc_prior(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # The weights should be a Dirichlet-normalized vector that sums to 1.
    weights = np.asarray(result["weights"], dtype=float)
    assert np.all(np.isfinite(weights))
    assert abs(float(np.sum(weights)) - 1.0) < 1e-8
    # Merging the first two weights must reproduce the consistency gap.
    merged_first_two = weights[0] + weights[1]
    expected_gap = abs(merged_first_two + float(np.sum(weights[2:])) - 1.0)
    assert abs(float(result["aggregation_gap"]) - expected_gap) < 1e-12
    # Method tag per the literature reference in the docstring.
    assert result["method"].startswith("consistent finite-dimensional prior")


def test_gh_c3_2_edge():
    """Test edge case: a length-1 input clamps k to the documented minimum of 4."""
    result = ghosal_stochastic_proc_prior(np.array([42.0]))
    # With len(x)=1, the implementation must still return a Dirichlet of
    # size k = max(4, min(1, 8)) = 4.
    weights = np.asarray(result["weights"], dtype=float)
    assert weights.shape == (4,)
    assert abs(float(np.sum(weights)) - 1.0) < 1e-8
    # Aggregation-gap identity, recomputed independently.
    merged = weights[0] + weights[1]
    expected_gap = abs(merged + float(np.sum(weights[2:])) - 1.0)
    assert abs(float(result["aggregation_gap"]) - expected_gap) < 1e-12
    # Estimate is the first component of the weights vector.
    assert abs(float(result["estimate"]) - float(weights[0])) < 1e-12
