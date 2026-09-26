"""gccwv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gccwv import gccwv


def test_gccwv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gccwv()
