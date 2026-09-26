"""opmfo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opmfo import opmfo


def test_opmfo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opmfo()
