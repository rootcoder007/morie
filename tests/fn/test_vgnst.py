"""vgnst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgnst import vario_nested


def test_vgnst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_nested(coords=None, values=None)
