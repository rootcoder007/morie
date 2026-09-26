"""mpspl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpspl import mpspl


def test_mpspl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpspl()
