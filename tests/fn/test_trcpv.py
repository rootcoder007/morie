"""trcpv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trcpv import trcpv


def test_trcpv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trcpv()
