"""ctlvl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctlvl import ctlvl


def test_ctlvl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctlvl()
