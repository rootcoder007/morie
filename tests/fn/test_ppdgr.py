"""ppdgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppdgr import ppdgr


def test_ppdgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppdgr()
