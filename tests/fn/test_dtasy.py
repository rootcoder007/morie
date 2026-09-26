"""dtasy is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtasy import dtasy


def test_dtasy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtasy()
