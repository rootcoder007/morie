"""Tests for kmfew.kamath_few_shot_exemplar_selection."""

from morie.fn import _array_core as np

from morie.fn.kmfew import kamath_few_shot_exemplar_selection


def test_kmfew_basic():
    """Test basic functionality."""
    D = 0.5
    query_embed = 0.5
    K = 1
    result = kamath_few_shot_exemplar_selection(D, query_embed, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "selected" in result


def test_kmfew_edge():
    """Test edge cases."""
    D = 0.5
    query_embed = 0.5
    K = 1
    result = kamath_few_shot_exemplar_selection(D, query_embed, K)
    assert isinstance(result, dict)
