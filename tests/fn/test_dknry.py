"""dknry is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dknry import dknry


def test_dknry_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dknry()
