"""msdel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msdel import delaunay_2d


def test_msdel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        delaunay_2d(data=None)
