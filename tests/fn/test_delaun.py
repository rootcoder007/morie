"""delaun is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.delaun import delaun


def test_delaun_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        delaun()
