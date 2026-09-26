"""ubimv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubimv import ubimv


def test_ubimv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubimv()
