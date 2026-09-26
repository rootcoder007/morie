"""enbdv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enbdv import enbdv


def test_enbdv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enbdv()
