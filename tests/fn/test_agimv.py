"""agimv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agimv import agimv


def test_agimv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agimv()
