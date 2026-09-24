"""Tests for posterior_density_binomial.posterior_density_binomial."""

import math

from morie.fn import _array_core as np

from morie.fn.posterior_density_binomial import (
    posterior_density_binomial,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e23_basic():
    """Test basic functionality."""
    pi = 0.3
    w = 7
    n = 10
    a = 1.0
    b = 1.0
    result = posterior_density_binomial(pi, w, n, a, b)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e23_edge():
    """Test edge cases."""
    pi = 0.5
    w = 5
    n = 10
    a = 2.0
    b = 3.0
    result = posterior_density_binomial(pi, w, n, a, b)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
