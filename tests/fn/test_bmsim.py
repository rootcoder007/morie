"""bmsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bmsim import bmsim


def test_bmsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bmsim()
