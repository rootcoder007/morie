"""wqnh3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqnh3 import wqnh3


def test_wqnh3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqnh3()
