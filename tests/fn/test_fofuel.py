"""fofuel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fofuel import fofuel


def test_fofuel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fofuel()
