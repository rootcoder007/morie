"""Tests for levdis.levenshtein."""

import math

from morie.fn import _array_core as np

from morie.fn.levdis import levenshtein


def test_levdis_basic():
    """Test basic functionality."""
    s1 = list("kitten")
    s2 = list("sitting")
    result = levenshtein(s1, s2)
    assert math.isfinite(result)
    assert result == 3


def test_levdis_edge():
    """Test edge cases."""
    s1 = list("abc")
    s2 = list("abc")
    result = levenshtein(s1, s2)
    assert math.isfinite(result)
    assert result == 0
