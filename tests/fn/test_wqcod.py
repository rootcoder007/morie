"""wqcod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqcod import wqcod


def test_wqcod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqcod()
