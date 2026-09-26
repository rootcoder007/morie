"""ubfbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubfbr import ubfbr


def test_ubfbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubfbr()
