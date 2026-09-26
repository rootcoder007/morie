"""vmsil is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmsil import vmsil


def test_vmsil_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmsil()
