"""sfcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfcv import sfcv


def test_sfcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfcv(y=None, X=None, W=None)
