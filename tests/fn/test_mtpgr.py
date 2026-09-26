"""mtpgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtpgr import mtpgr


def test_mtpgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtpgr()
