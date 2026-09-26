"""ghtrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghtrd import ghtrd


def test_ghtrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghtrd()
