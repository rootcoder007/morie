"""wqtst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqtst import wqtst


def test_wqtst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqtst()
