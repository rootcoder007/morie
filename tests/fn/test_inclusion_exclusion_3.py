"""Tests for inclusion_exclusion_3.inclusion_exclusion_3."""

import math

from morie.fn import _array_core as np

from morie.fn.inclusion_exclusion_3 import (
    inclusion_exclusion_3,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e92_basic():
    """Test basic functionality."""
    p_a, p_b, p_c = 0.4, 0.5, 0.6
    p_ab, p_ac, p_bc = 0.2, 0.25, 0.3
    p_abc = 0.1
    result = inclusion_exclusion_3(p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc)
    assert hasattr(result, "payload")
    assert "p_or" in result.payload
    assert math.isfinite(result.payload["p_or"])
    assert 0.0 <= result.payload["p_or"] <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e92_edge():
    """Test edge cases."""
    p_a, p_b, p_c = 0.1, 0.2, 0.15
    p_ab, p_ac, p_bc = 0.02, 0.015, 0.03
    p_abc = 0.005
    result = inclusion_exclusion_3(p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc)
    assert hasattr(result, "payload")
    assert "p_or" in result.payload
    assert math.isfinite(result.payload["p_or"])
    assert 0.0 <= result.payload["p_or"] <= 1.0
