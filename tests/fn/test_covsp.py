"""Tests for covsp.one_sample_coverage.

The generator guessed an ``estimate`` key this function never returns;
the real payload is the per-gap coverage vector plus its cumulative and
expected values. Expectations below are derived from the definition,
not from running the function: for n distinct order statistics the
sample splits the line into n + 1 gaps, each of expected coverage
1 / (n + 1) under a continuous distribution.
"""

import numpy as np

from morie.fn.covsp import one_sample_coverage


def test_covsp_basic():
    """Coverage vector has n + 1 gaps, each with expectation 1/(n+1)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = one_sample_coverage(x)
    cov = np.asarray(result["coverages"], dtype=float)
    assert cov.size == x.size + 1
    assert np.all(np.isfinite(cov))
    assert result["expected"] == 1.0 / (x.size + 1)
    # equally spaced data splits its interior evenly
    assert np.allclose(cov, 1.0 / (x.size + 1))
    assert result["n"] == x.size
    assert result["sample_min"] == 1.0
    assert result["sample_max"] == 5.0


def test_covsp_cumulative_is_interior_mass():
    """Cumulative coverage counts the interior gaps only: (n-1)/(n+1)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = one_sample_coverage(x)
    assert np.isclose(result["cumulative"], (x.size - 1) / (x.size + 1))


def test_covsp_edge():
    """n < 2 is the documented degenerate branch: no gaps, nan summaries."""
    result = one_sample_coverage(np.array([42.0]))
    assert result["n"] == 1
    assert np.asarray(result["coverages"]).size == 0
    assert np.isnan(result["cumulative"])
    assert np.isnan(result["expected"])
    assert "sample_min" not in result
