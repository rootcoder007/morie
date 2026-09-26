"""svpnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svpnl import svpnl


def test_svpnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svpnl()
