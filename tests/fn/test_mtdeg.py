"""mtdeg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtdeg import mtdeg


def test_mtdeg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtdeg()
