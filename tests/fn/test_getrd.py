"""getrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.getrd import getrd


def test_getrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        getrd()
