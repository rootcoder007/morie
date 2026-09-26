"""zsvor re-exports the real voronoi_areas from voron."""

from morie.fn.voron import voronoi_areas as canonical
from morie.fn.zsvor import voronoi_areas


def test_zsvor_is_the_canonical_implementation():
    assert voronoi_areas is canonical
