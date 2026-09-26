"""curvgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.curvgr import curvgr


def test_curvgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        curvgr()
