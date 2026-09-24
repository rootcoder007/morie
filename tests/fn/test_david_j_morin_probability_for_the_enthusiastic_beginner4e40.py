"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner4e40.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e40 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e40_basic():
    """Test basic functionality."""
    import warnings

    rng = np.random.default_rng(42)
    k = int(rng.integers(0, 20))
    a = 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40(k, a)
    assert isinstance(result, dict)
    assert "probability" in result
    p = float(result["probability"])
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e40_edge():
    """Test edge cases."""
    import warnings

    k = 0
    a = 2.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_40(k, a)
    assert isinstance(result, dict)
    assert "probability" in result
    p = float(result["probability"])
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
