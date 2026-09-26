"""wlsrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlsrv import wlsrv


def test_wlsrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlsrv()
