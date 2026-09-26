"""geloc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geloc import geloc


def test_geloc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geloc()
