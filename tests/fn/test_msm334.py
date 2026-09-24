"""Tests for msm334.mvsml_preprocessing_eq_2_22."""

from morie.fn import _array_core as np

from morie.fn.msm334 import mvsml_preprocessing_eq_2_22


def test_msm334_basic():
    """Test basic functionality."""
    sigma2 = 1.0
    x_star = [1.0, 1.0]
    eigenvalues = [4.0, 4.0]
    result = mvsml_preprocessing_eq_2_22(sigma2, x_star, eigenvalues)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm334_edge():
    """Test edge cases."""
    sigma2 = 1.0
    x_star = [1.0, 1.0]
    eigenvalues = [4.0, 4.0]
    result = mvsml_preprocessing_eq_2_22(sigma2, x_star, eigenvalues)
    assert isinstance(result, dict)
