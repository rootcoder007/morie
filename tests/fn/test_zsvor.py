"""zsvor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsvor import voronoi_areas


def test_zsvor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        voronoi_areas(data=None)
