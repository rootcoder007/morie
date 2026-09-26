"""zsgpv re-exports the real gp_variance from gpvarF."""

from morie.fn.gpvarF import gp_variance as canonical
from morie.fn.zsgpv import gp_variance


def test_zsgpv_is_the_canonical_implementation():
    assert gp_variance is canonical
