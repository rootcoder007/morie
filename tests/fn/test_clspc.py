"""clspc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clspc import clspc


def test_clspc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clspc()
