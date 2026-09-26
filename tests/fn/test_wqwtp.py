"""wqwtp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqwtp import wqwtp


def test_wqwtp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqwtp()
