"""dkchg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkchg import dkchg


def test_dkchg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkchg()
