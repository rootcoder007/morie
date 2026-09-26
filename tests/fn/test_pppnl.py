"""pppnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pppnl import pppnl


def test_pppnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pppnl()
