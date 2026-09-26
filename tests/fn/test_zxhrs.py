"""zxhrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxhrs import hier_spatial_fe


def test_zxhrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hier_spatial_fe(data=None)
