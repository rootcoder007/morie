"""gcdox is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcdox import gcdox


def test_gcdox_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcdox()
