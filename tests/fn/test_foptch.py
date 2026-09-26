"""foptch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foptch import foptch


def test_foptch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foptch()
