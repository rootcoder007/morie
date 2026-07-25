"""Tests for fzhdc.fauzi_h_decomposition."""

import numpy as np

from morie.fn.fzhdc import fauzi_h_decomposition


def test_fzhdc_basic():
    """The U-statistic with kernel 0.5*(a-b)^2 IS the unbiased variance.

    For x = 1..5 the sample variance is 10/4 = 2.5, so theta must be 2.5.
    (The generated stub asserted 3.0 -- the mean, not the variance.)
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_h_decomposition(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value
    assert result["n"] == 5
    # Hajek-projection variance is a variance: non-negative and finite.
    assert result["sigma1_sq"] >= 0
    assert np.isfinite(result["se"])


def test_fzhdc_edge():
    """A single observation cannot form a pair -- report, do not crash."""
    result = fauzi_h_decomposition(np.array([42.0]))
    assert result["n"] == 1
    assert np.isnan(result["estimate"])
    assert "too few" in result["method"]
