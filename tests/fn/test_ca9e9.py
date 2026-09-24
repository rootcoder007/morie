"""Tests for ca9e9.ca_chapter_9_equation_9."""

import math

from morie.fn import _array_core as np

from morie.fn.ca9e9 import ca_chapter_9_equation_9


def test_ca9e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b = 3
    groups = [rng.normal(0, 1, (5, b)), rng.normal(0, 1, (5, b))]
    result = ca_chapter_9_equation_9(groups)
    assert isinstance(result, dict)
    assert "ms_b_subjects" in result
    assert math.isfinite(result["ms_b_subjects"])


def test_ca9e9_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    b = 2
    groups = [
        rng.normal(0, 1, (2, b)),
        rng.normal(0, 1, (2, b)),
        rng.normal(0, 1, (2, b)),
    ]
    result = ca_chapter_9_equation_9(groups)
    assert isinstance(result, dict)
    assert "ms_b_subjects" in result
    assert math.isfinite(result["ms_b_subjects"])
