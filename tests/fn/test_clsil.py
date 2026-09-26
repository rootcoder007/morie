"""clsil is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clsil import clsil


def test_clsil_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clsil()
