"""rsmsv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsmsv import rsmsv


def test_rsmsv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsmsv()
