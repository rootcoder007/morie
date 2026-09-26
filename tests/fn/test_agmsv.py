"""agmsv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agmsv import agmsv


def test_agmsv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agmsv()
