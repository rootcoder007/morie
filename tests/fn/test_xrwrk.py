"""xrwrk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwrk import w_rook


def test_xrwrk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_rook(data=None)
