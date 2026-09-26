"""hsasy is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsasy import hsasy


def test_hsasy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsasy()
