"""wqno3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqno3 import wqno3


def test_wqno3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqno3()
