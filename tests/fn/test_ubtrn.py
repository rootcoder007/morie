"""ubtrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubtrn import ubtrn


def test_ubtrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubtrn()
