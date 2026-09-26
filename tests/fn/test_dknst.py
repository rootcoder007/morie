"""dknst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dknst import dknst


def test_dknst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dknst()
