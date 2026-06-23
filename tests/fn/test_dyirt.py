"""Test dyirt."""

from morie.fn.dyirt import dynamic_irt_estimate


def test_dyirt_basic():
    r = dynamic_irt_estimate()
    assert r.value is not None


def test_dyirt_name():
    r = dynamic_irt_estimate()
    assert r.name
