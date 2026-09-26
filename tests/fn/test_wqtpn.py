"""wqtpn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtpn import wqtpn


def test_wqtpn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtpn()
