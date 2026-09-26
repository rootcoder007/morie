"""rfstud is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfstud import rfstud


def test_rfstud_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfstud()
