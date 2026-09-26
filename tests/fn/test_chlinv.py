"""chlinv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlinv import chlinv


def test_chlinv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlinv()
