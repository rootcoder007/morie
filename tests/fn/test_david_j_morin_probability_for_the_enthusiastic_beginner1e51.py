"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e51.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51."""

import warnings

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e51 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e51_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = int(rng.integers(0, 10))
    N = int(rng.integers(2, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51(n, N)
    result_repr = repr(result)
    assert f"n    {n}" in result_repr
    assert f"N    {N}" in result_repr


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e51_edge():
    """Test edge cases."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_51(0, 2)
    result_repr = repr(result)
    assert "n    0" in result_repr
    assert "N    2" in result_repr
