"""Tests for ca7e3.ca_chapter_7_equation_3."""

import math

from morie.fn import _array_core as np

from morie.fn.ca7e3 import ca_chapter_7_equation_3


def test_ca7e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    beta0 = 1.0
    u_j = float(rng.normal(0, 1))
    e_ij = float(rng.normal(0, 1))
    result = ca_chapter_7_equation_3(beta0, u_j, e_ij)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca7e3_edge():
    """Test edge cases."""
    result = ca_chapter_7_equation_3(0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
