"""Tests for ca9e8.ca_chapter_9_equation_8."""

import math

from morie.fn import _array_core as np

from morie.fn.ca9e8 import ca_chapter_9_equation_8


def test_ca9e8_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    a, n, b = 3, 5, 4
    groups = [rng.normal(0, 1, (n, b)) for _ in range(a)]
    result = ca_chapter_9_equation_8(groups)
    assert isinstance(result, dict)
    assert "ms_subjects" in result
    assert math.isfinite(float(result["ms_subjects"]))


def test_ca9e8_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    a, n, b = 2, 3, 2
    groups = [rng.normal(0, 1, (n, b)) for _ in range(a)]
    result = ca_chapter_9_equation_8(groups)
    assert isinstance(result, dict)
    assert "ms_subjects" in result
    assert math.isfinite(float(result["ms_subjects"]))
