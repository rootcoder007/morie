"""dkinl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkinl import dkinl


def test_dkinl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkinl()
