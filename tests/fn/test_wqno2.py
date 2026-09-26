"""wqno2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqno2 import wqno2


def test_wqno2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqno2()
