"""Tests for bivand20137e2.bivand2013_chapter_7_equation_2."""

from morie.fn import _array_core as np

import pytest

from morie.fn.bivand20137e2 import bivand2013_chapter_7_equation_2


def test_bivand20137e2_basic():
    """Test that the function raises NotImplementedError (extractor artefact)."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    with pytest.raises(NotImplementedError):
        bivand2013_chapter_7_equation_2(x)


def test_bivand20137e2_edge():
    """Test edge cases: still raises NotImplementedError."""
    x = np.random.default_rng(42).normal(0, 1, 10)
    with pytest.raises(NotImplementedError):
        bivand2013_chapter_7_equation_2(x)
