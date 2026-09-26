"""ppcnv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppcnv import ppcnv


def test_ppcnv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppcnv()
