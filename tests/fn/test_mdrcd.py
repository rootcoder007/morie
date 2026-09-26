"""mdrcd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdrcd import mdrcd


def test_mdrcd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdrcd()
