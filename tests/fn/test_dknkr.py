"""dknkr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dknkr import dknkr


def test_dknkr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dknkr()
