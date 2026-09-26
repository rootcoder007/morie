"""wlcmr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlcmr import wlcmr


def test_wlcmr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlcmr()
