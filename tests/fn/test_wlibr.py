"""wlibr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlibr import wlibr


def test_wlibr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlibr()
