"""wlrst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlrst import wlrst


def test_wlrst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlrst()
