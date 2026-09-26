"""gccnv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gccnv import gccnv


def test_gccnv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gccnv()
