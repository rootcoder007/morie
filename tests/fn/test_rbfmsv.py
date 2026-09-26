"""rbfmsv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rbfmsv import rbfmsv


def test_rbfmsv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rbfmsv()
