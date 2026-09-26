"""wlfst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlfst import wlfst


def test_wlfst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlfst()
