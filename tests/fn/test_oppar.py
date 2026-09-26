"""oppar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.oppar import oppar


def test_oppar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oppar()
