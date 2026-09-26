"""dtcpg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcpg import dtcpg


def test_dtcpg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcpg()
