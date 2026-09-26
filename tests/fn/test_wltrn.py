"""wltrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wltrn import wltrn


def test_wltrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wltrn()
