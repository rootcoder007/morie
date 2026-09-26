"""xrwqn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwqn import w_queen


def test_xrwqn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_queen(data=None)
