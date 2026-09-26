"""vmspc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vmspc import vmspc


def test_vmspc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vmspc()
