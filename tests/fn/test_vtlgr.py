"""vtlgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtlgr import vtlgr


def test_vtlgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtlgr()
