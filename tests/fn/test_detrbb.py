"""Tests for detrbb.detr_set_prediction."""

import math

from morie.fn import _array_core as np

from morie.fn.detrbb import detr_set_prediction


def test_detrbb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    Q, G = 5, 3
    image = rng.uniform(0, 1, (Q, 4))
    queries = rng.uniform(0, 1, (G, 4))
    result = detr_set_prediction(image, queries, G)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "assignment" in result
    assert "cost" in result
    assert "matched" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["cost"])
    assert len(result["assignment"]) == G
    assert len(result["matched"]) == G


def test_detrbb_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    Q, G = 2, 1
    image = rng.uniform(0, 1, (Q, 4))
    queries = rng.uniform(0, 1, (G, 4))
    result = detr_set_prediction(image, queries)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "assignment" in result
    assert "cost" in result
    assert "matched" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["cost"])
    assert len(result["assignment"]) == G
    assert len(result["matched"]) == G
