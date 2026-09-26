"""unstgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.unstgr import unstgr


def test_unstgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        unstgr()
