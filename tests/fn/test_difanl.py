"""difanl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.difanl import difanl


def test_difanl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        difanl()
