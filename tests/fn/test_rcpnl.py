"""rcpnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcpnl import rcpnl


def test_rcpnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcpnl()
