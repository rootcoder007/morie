"""clxbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clxbr import clxbr


def test_clxbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clxbr()
