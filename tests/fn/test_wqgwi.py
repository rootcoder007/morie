"""wqgwi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqgwi import wqgwi


def test_wqgwi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqgwi()
