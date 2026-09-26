"""opga is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opga import opga


def test_opga_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opga()
