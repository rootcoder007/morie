"""Tests for alann.alammar_approximate_nearest_neighbor."""

from morie.fn.alann import alammar_approximate_nearest_neighbor


def test_alann_basic():
    pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    idx = {"points": pts, "neighbors": [[1], [0, 2], [1]], "entry": 0}
    out = alammar_approximate_nearest_neighbor([1.9, 0.0], idx)
    assert out["nearest"] == 2
    assert out["found_exact"] is True


def test_alann_edge():
    import pytest
    pts = [[0.0], [1.0]]
    with pytest.raises(ValueError, match="entry point"):
        alammar_approximate_nearest_neighbor([0.0],
            {"points": pts, "neighbors": [[1], [0]], "entry": 5})
