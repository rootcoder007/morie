"""Tests for ca8e4.ca_chapter_8_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.ca8e4 import ca_chapter_8_equation_4


def test_ca8e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    sigma_means = rng.uniform(0.1, 5.0)
    sigma_error = rng.uniform(0.1, 5.0)
    result = ca_chapter_8_equation_4(sigma_means, sigma_error)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca8e4_edge():
    """Test edge cases."""
    sigma_means = 0.5
    sigma_error = 0.5
    result = ca_chapter_8_equation_4(sigma_means, sigma_error)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
