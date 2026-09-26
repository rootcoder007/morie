"""wlocc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlocc import wlocc


def test_wlocc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlocc()
