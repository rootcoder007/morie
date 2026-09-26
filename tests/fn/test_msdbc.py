"""msdbc re-exports the real double_center from hmmds."""

from morie.fn.hmmds import double_center as canonical
from morie.fn.msdbc import double_center


def test_msdbc_is_the_canonical_implementation():
    assert double_center is canonical
