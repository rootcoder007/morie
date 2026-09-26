"""isval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isval import isval


def test_isval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isval()
