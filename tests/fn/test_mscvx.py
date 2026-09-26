"""mscvx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mscvx import convex_hull_2d


def test_mscvx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        convex_hull_2d(data=None)
