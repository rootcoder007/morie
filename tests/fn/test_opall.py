"""opall is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opall import opall


def test_opall_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opall()
