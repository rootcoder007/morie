"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner6e76.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_76."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e76 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_76,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e76_basic():
    """Test basic functionality with default-like scalars."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_76(
        m=1.0, sigma_x=7.5, sigma_z=10.6
    )
    # The wrapper delegates to linmodel, which returns mu_y, sigma_y, r.
    # Accept either a dict/RichResult or a 3-sequence.
    if isinstance(result, dict):
        assert "mu_y" in result
        assert "sigma_y" in result
        assert "r" in result
        values = [result["mu_y"], result["sigma_y"], result["r"]]
    else:
        assert len(result) == 3
        mu_y, sigma_y, r = result
        values = [mu_y, sigma_y, r]
    for v in values:
        assert math.isfinite(float(v))


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e76_edge():
    """Test edge case with different valid scalar parameters."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_76(
        m=0.5, sigma_x=1.0, sigma_z=2.0
    )
    if isinstance(result, dict):
        assert "mu_y" in result
        assert "sigma_y" in result
        assert "r" in result
        values = [result["mu_y"], result["sigma_y"], result["r"]]
    else:
        assert len(result) == 3
        mu_y, sigma_y, r = result
        values = [mu_y, sigma_y, r]
    for v in values:
        assert math.isfinite(float(v))
