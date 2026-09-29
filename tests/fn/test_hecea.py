"""Tests for morie.fn.hecea: values recomputed from the definition."""

from morie.fn.hecea import cost_effectiveness_plane


def test_quadrant_percentages():
    c = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0]
    e = [0.1, 0.2, -0.3, 0.4, -0.5, 0.0]
    r = cost_effectiveness_plane(c, e)
    q = r.value
    assert abs(q["NE"] - 100 * sum(1 for a, b in zip(c, e) if a > 0 and b > 0) / 6) < 1e-12
    assert abs(q["NW"] - 100 * sum(1 for a, b in zip(c, e) if a > 0 and b <= 0) / 6) < 1e-12
    assert abs(sum(q.values()) - 100.0) < 1e-12
