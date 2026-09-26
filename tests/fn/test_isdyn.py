"""isdyn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isdyn import isdyn


def test_isdyn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isdyn()
