"""srlgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srlgr import srlgr


def test_srlgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srlgr()
