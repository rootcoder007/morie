"""lvsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lvsim import lvsim


def test_lvsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lvsim()
