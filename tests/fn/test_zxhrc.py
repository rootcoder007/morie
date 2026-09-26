"""zxhrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxhrc import hier_spatial_cross


def test_zxhrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hier_spatial_cross(data=None)
