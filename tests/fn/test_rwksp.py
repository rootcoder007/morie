"""rwksp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rwksp import rwksp


def test_rwksp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rwksp()
