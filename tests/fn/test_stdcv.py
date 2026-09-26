"""stdcv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stdcv import stdcv


def test_stdcv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stdcv()
