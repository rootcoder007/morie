"""mtmlr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtmlr import mtmlr


def test_mtmlr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtmlr()
