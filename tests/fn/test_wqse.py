"""wqse is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqse import wqse


def test_wqse_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqse()
