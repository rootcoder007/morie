"""wllcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wllcv import wllcv


def test_wllcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wllcv()
