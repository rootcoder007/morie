"""icc31c re-exports the real icc_one_way from icc1."""

from morie.fn.icc1 import icc_one_way as canonical
from morie.fn.icc31c import icc_one_way


def test_icc31c_is_the_canonical_implementation():
    assert icc_one_way is canonical
