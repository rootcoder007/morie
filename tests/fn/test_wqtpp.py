"""wqtpp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtpp import wqtpp


def test_wqtpp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtpp()
