"""wqdrk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqdrk import wqdrk


def test_wqdrk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqdrk()
