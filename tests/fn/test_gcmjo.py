"""gcmjo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcmjo import gcmjo


def test_gcmjo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcmjo()
