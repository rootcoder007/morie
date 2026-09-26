"""enlgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enlgr import enlgr


def test_enlgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enlgr()
