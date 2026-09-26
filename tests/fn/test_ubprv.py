"""ubprv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubprv import ubprv


def test_ubprv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubprv()
