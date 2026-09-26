"""zxpos is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxpos import possibilistic_sp


def test_zxpos_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        possibilistic_sp(data=None)
