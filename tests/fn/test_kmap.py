"""Tests for kmap.kamath_autoprompt_gradient_search."""

from morie.fn import _array_core as np

from morie.fn.kmap import kamath_autoprompt_gradient_search


def test_kmap_basic():
    """Test basic functionality."""
    template = [None, 'y']
    dataset = [1]
    model = lambda tpl, d: 0.0
    result = kamath_autoprompt_gradient_search(template, dataset, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmap_edge():
    """Test edge cases."""
    template = [None, 'y']
    dataset = [1]
    model = lambda tpl, d: 0.0
    result = kamath_autoprompt_gradient_search(template, dataset, model)
    assert isinstance(result, dict)
