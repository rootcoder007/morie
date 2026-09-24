"""Tests for semsro.sem_residual."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.semsro import sem_residual


def _random_symmetric_cov(rng, p, diag_lo=0.5, diag_hi=1.5, off_std=0.3):
    """Generate a symmetric matrix with positive diagonal entries."""
    cov = [[0.0] * p for _ in range(p)]
    for i in range(p):
        cov[i][i] = rng.uniform(diag_lo, diag_hi)
        for j in range(i):
            cov[i][j] = rng.normal(0, off_std)
            cov[j][i] = cov[i][j]
    return cov


def test_semsro_basic():
    """Test basic functionality with a well-formed covariance pair."""
    rng = np.random.default_rng(42)
    p = 3
    sample_cov = _random_symmetric_cov(rng, p)
    rng_fitted = np.random.default_rng(43)
    fitted_cov = _random_symmetric_cov(rng_fitted, p)

    result = sem_residual(sample_cov, fitted_cov)

    # Result is a RichResult (dict-like) containing specific keys
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "residual" in result
    assert "max_abs_residual" in result

    # SRMR is a finite, non-negative scalar
    srmr = result["estimate"]
    assert isinstance(srmr, float)
    assert math.isfinite(srmr)
    assert srmr >= 0.0

    # Residual matrix has shape (p, p)
    residual = result["residual"]
    assert isinstance(residual, list)
    assert len(residual) == p
    for row in residual:
        assert isinstance(row, list)
        assert len(row) == p

    # max_abs_residual is finite and non-negative
    max_abs = result["max_abs_residual"]
    assert isinstance(max_abs, float)
    assert math.isfinite(max_abs)
    assert max_abs >= 0.0


def test_semsro_edge():
    """Test that a non-symmetric sample_cov raises ValueError as documented."""
    # Non-symmetric 2x2 matrix
    sample_cov = [[1.0, 0.5],
                  [0.6, 1.0]]
    fitted_cov = [[1.0, 0.0],
                  [0.0, 1.0]]

    with pytest.raises(ValueError):
        sem_residual(sample_cov, fitted_cov)
