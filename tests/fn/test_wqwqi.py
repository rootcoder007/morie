"""wqwqi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqwqi import wqwqi


def test_wqwqi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqwqi()
