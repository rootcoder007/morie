"""ptvor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptvor import pp_voronoi


def test_ptvor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pp_voronoi(data=None)
