"""rsfcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsfcv import rsfcv


def test_rsfcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsfcv()
