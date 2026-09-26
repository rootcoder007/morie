"""vgani re-exports the real anisotropy_ratio from sganr."""

from morie.fn.sganr import anisotropy_ratio as canonical
from morie.fn.vgani import anisotropy_ratio


def test_vgani_is_the_canonical_implementation():
    assert anisotropy_ratio is canonical
