"""dkmvv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkmvv import dkmvv


def test_dkmvv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkmvv()
