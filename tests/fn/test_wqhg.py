"""wqhg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqhg import wqhg


def test_wqhg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqhg()
