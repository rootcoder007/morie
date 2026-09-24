"""Tests for beta_pdf.beta_pdf."""

from morie.fn import _array_core as np

from morie.fn.beta_pdf import (
    beta_pdf,
)

import math


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    v = float(rng.uniform(0, 1))
    result = beta_pdf(v, a=2, b=5)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    val = result["value"]
    assert math.isfinite(val)
    assert val >= 0.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e5_edge():
    """Test edge cases."""
    # Edge: symmetric Beta(2,2) at v=0.5
    result = beta_pdf(0.5, a=2, b=2)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    val = result["value"]
    assert math.isfinite(val)
    assert val >= 0.0
