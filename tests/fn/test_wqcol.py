"""wqcol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqcol import wqcol


def test_wqcol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqcol()
