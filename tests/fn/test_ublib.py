"""ublib is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ublib import ublib


def test_ublib_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ublib()
