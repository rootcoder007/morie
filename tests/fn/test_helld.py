"""Tests for hellinger_dist."""

from morie.fn.helld import helld, hellinger_dist


def test_same():
    r = hellinger_dist([0.5, 0.5], [0.5, 0.5])
    assert abs(r.estimate) < 1e-10


def test_different():
    r = hellinger_dist([1, 0], [0, 1])
    assert abs(r.estimate - 1.0) < 1e-10


def test_alias():
    assert helld is hellinger_dist
