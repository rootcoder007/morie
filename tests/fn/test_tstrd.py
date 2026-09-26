"""tstrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tstrd import tstrd


def test_tstrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tstrd()
