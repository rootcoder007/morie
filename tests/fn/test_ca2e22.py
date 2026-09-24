"""Tests for ca2e22.ca_chapter_2_equation_22."""

from morie.fn import _array_core as np

from morie.fn.ca2e22 import ca_chapter_2_equation_22

import math
import pytest

def test_ca2e22_basic():
    """Test basic functionality."""
    b0 = 2.0
    bs = [0.5, 0.3, 0.2]
    dummy_index = 0
    result = ca_chapter_2_equation_22(b0, bs, dummy_index)
    assert isinstance(result, dict)
    assert "intercept" in result
    assert math.isclose(result["intercept"], b0 + bs[dummy_index])
    assert "method" in result

def test_ca2e22_edge():
    """Test edge cases."""
    b0 = 0.0
    bs = [0.0]
    dummy_index = 0
    result = ca_chapter_2_equation_22(b0, bs, dummy_index)
    assert isinstance(result, dict)
    assert "intercept" in result
    assert math.isclose(result["intercept"], 0.0)
