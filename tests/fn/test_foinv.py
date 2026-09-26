"""foinv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foinv import foinv


def test_foinv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foinv()
