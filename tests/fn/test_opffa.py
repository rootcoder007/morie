"""opffa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opffa import opffa


def test_opffa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opffa()
