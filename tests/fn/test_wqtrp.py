"""wqtrp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtrp import wqtrp


def test_wqtrp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtrp()
