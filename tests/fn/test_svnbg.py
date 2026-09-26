"""svnbg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svnbg import nash_bargain_sp


def test_svnbg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nash_bargain_sp(data=None)
