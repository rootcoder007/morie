"""opdbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opdbr import opdbr


def test_opdbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opdbr()
