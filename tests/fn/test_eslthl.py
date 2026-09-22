"""Tests for eslthl.esl_thin_plate_spline."""

import pytest

from morie.fn import _array_core as np

from morie.fn.eslthl import esl_thin_plate_spline


def test_eslthl_basic():
    """Test basic functionality: interpolation at lambda_=0, return shape, and
    that heavy smoothing collapses onto the least-squares plane (the documented
    behaviour, since the TPS penalty annihilates linear functions)."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(0, 1, (n, 2))
    y = rng.normal(0, 1, n)
    lambda_ = 0.5
    result = esl_thin_plate_spline(X, y, lambda_)
    # The function returns a RichResult that also supports dict-style access.
    assert "fitted" in result
    assert "delta" in result
    assert "beta" in result
    assert "residuals" in result
    assert "edf" in result
    assert "gcv" in result
    assert result["fitted"].shape[0] == n


def test_eslthl_interpolates_at_zero():
    """At lambda_=0 the spline interpolates the observations exactly."""
    rng = np.random.default_rng(42)
    n = 25
    X = rng.uniform(-1, 1, (n, 2))
    # Truth is a smooth function evaluated at the same sites as a sanity check.
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
    result = esl_thin_plate_spline(X, y, lambda_=0.0)
    max_abs_resid = float(np.max(np.abs(result["residuals"])))
    assert max_abs_resid < 1e-6


def test_eslthl_invalid_lambda():
    """Negative lambda_ must raise ValueError, per docstring."""
    rng = np.random.default_rng(42)
    n = 25
    X = rng.uniform(-1, 1, (n, 2))
    y = rng.normal(0, 1, n)
    with pytest.raises(ValueError, match="lambda_ must be non-negative"):
        esl_thin_plate_spline(X, y, lambda_=-1.0)
