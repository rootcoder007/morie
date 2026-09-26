"""plcrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plcrd import plcrd


def test_plcrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plcrd()
