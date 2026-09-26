"""wqpo4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqpo4 import wqpo4


def test_wqpo4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqpo4()
