"""mhst1 re-exports the real mantel_haenszel_or from mhors."""

from morie.fn.mhors import mantel_haenszel_or as canonical
from morie.fn.mhst1 import mantel_haenszel_or


def test_mhst1_is_the_canonical_implementation():
    assert mantel_haenszel_or is canonical
