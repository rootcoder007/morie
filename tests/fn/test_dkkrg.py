"""dkkrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkkrg import dkkrg


def test_dkkrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkkrg()
