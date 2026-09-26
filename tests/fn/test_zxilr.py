"""zxilr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxilr import ilr_spatial


def test_zxilr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ilr_spatial(data=None)
