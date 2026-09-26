"""wqhrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqhrd import wqhrd


def test_wqhrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqhrd()
