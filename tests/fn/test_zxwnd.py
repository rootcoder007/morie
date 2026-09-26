"""zxwnd re-exports the real wind_rose from winros."""

from morie.fn.winros import wind_rose as canonical
from morie.fn.zxwnd import wind_rose


def test_zxwnd_is_the_canonical_implementation():
    assert wind_rose is canonical
