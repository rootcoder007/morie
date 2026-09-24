"""Tests for classify_events.classify_events."""

from morie.fn import _array_core as np

from morie.fn.classify_events import (
    classify_events,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e24_basic():
    """Test basic functionality."""
    # Independent events: p_ab == p_a * p_b
    p_a, p_b, p_ab = 0.5, 0.4, 0.2
    result = classify_events(p_a, p_b, p_ab)
    assert isinstance(result, dict)
    assert result["independent"] is True
    assert result["exclusive"] is False
    assert result["p_a"] == float(p_a)
    assert result["p_b"] == float(p_b)
    assert result["p_ab"] == float(p_ab)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e24_edge():
    """Test edge cases."""
    # Exclusive events: p_ab == 0
    p_a, p_b, p_ab = 0.3, 0.5, 0.0
    result = classify_events(p_a, p_b, p_ab)
    assert isinstance(result, dict)
    assert result["independent"] is False
    assert result["exclusive"] is True
    assert result["p_a"] == float(p_a)
    assert result["p_b"] == float(p_b)
    assert result["p_ab"] == float(p_ab)
