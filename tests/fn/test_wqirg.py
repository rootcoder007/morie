"""wqirg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqirg import wqirg


def test_wqirg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqirg()
