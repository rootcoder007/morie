"""Tests for morie.fn.smean: recompute from the definition."""

from morie.fn.smean import sample_mean

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_mean():
    assert abs(sample_mean(X).value - sum(X) / len(X)) < 1e-15
    assert sample_mean(X).extra["n"] == 8
