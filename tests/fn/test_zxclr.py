"""zxclr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxclr import clr_spatial


def test_zxclr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clr_spatial(data=None)
