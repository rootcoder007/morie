"""vtblk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtblk import vtblk


def test_vtblk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtblk()
