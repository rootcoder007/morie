"""isatt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isatt import isatt


def test_isatt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isatt()
