"""ubcon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcon import ubcon


def test_ubcon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcon()
