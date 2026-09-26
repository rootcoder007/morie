"""clreg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clreg import clreg


def test_clreg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clreg()
