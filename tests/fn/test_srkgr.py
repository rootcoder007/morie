"""srkgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srkgr import srkgr


def test_srkgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srkgr()
