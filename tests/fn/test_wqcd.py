"""wqcd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqcd import wqcd


def test_wqcd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqcd()
