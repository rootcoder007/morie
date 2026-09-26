"""opskr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opskr import opskr


def test_opskr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opskr()
