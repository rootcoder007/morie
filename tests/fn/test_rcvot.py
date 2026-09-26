"""rcvot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcvot import rcvot


def test_rcvot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcvot()
