"""Tests for ca9e12.ca_chapter_9_equation_12."""

import numpy as _np

from morie.fn import _array_core as np

from morie.fn.ca9e12 import ca_chapter_9_equation_12


def test_ca9e12_basic():
    """Test basic functionality with two or more groups of >= 2 obs each."""
    rng = np.random.default_rng(42)
    # ANOVA needs >= 2 groups, each with >= 2 observations.
    group1 = rng.normal(0.0, 1.0, 10)
    group2 = rng.normal(0.5, 1.0, 10)
    groups = [group1, group2]
    result = ca_chapter_9_equation_12(groups)
    assert isinstance(result, dict)
    # Headline key is 'f' (the F-statistic), per the docstring.
    assert "f" in result
    assert "method" in result


def test_ca9e12_edge():
    """Test edge cases: minimum viable design (2 groups, 2 obs each)."""
    rng = np.random.default_rng(42)
    group1 = rng.normal(0.0, 1.0, 2)
    group2 = rng.normal(0.0, 1.0, 2)
    groups = [group1, group2]
    result = ca_chapter_9_equation_12(groups)
    assert isinstance(result, dict)
    assert "f" in result
