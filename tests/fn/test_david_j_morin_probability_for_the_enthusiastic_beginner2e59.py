"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner2e59.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59."""

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner2e59 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59,
)

import math


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e59_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p_a = float(rng.uniform(0.1, 0.9))
    p_z_given_a = float(rng.uniform(0.1, 0.9))
    p_z_given_not_a = float(rng.uniform(0.1, 0.9))
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59(
        p_a, p_z_given_a, p_z_given_not_a
    )
    # The function returns a RichResult; extract the posterior probability.
    assert isinstance(result, dict)
    assert "posterior" in result
    posterior = result["posterior"]
    # The posterior must be a finite probability between 0 and 1.
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e59_edge():
    """Test edge cases."""
    # Prior probability 1 -> posterior must be 1.
    p_a = 1.0
    p_z_given_a = 0.4
    p_z_given_not_a = 0.7
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59(
        p_a, p_z_given_a, p_z_given_not_a
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    posterior = result["posterior"]
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0
    assert posterior == 1.0

    # Prior probability 0 -> posterior must be 0.
    p_a = 0.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59(
        p_a, p_z_given_a, p_z_given_not_a
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    posterior = result["posterior"]
    assert math.isfinite(posterior)
    assert 0.0 <= posterior <= 1.0
    assert posterior == 0.0
