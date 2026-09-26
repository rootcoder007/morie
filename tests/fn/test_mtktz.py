"""mtktz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtktz import mtktz


def test_mtktz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtktz()
