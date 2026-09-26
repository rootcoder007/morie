"""gppost is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gppost import gppost


def test_gppost_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gppost()
