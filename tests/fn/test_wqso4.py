"""wqso4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqso4 import wqso4


def test_wqso4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqso4()
