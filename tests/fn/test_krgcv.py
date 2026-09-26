"""krgcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.krgcv import krgcv


def test_krgcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        krgcv()
