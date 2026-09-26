"""wlcrr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlcrr import wlcrr


def test_wlcrr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlcrr()
