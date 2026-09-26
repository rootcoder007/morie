"""ubcng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcng import ubcng


def test_ubcng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcng()
