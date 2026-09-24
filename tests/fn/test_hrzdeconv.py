"""Tests for hrzdeconv.horowitz_deconvolution_density."""

import pytest
from morie.fn import _array_core as np
from morie.fn.hrzdeconv import horowitz_deconvolution_density


def test_hrzdeconv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(45)
    w = rng.normal(0, 1, 100)
    sigma_eps = 0.5
    bandwidth = 0.3
    with pytest.raises(TypeError):
        horowitz_deconvolution_density(w, sigma_eps, bandwidth)


def test_hrzdeconv_edge():
    """Test edge cases."""
    rng = np.random.default_rng(45)
    w = rng.normal(0, 1, 100)
    sigma_eps = 0.5
    bandwidth = 0.3
    with pytest.raises(TypeError):
        horowitz_deconvolution_density(w, sigma_eps, bandwidth)
