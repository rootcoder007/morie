"""tssop is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssop import tssop


def test_tssop_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssop()
