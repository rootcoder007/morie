"""pprip is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pprip import pprip


def test_pprip_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pprip()
