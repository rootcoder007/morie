"""dtcpg2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcpg2 import dtcpg2


def test_dtcpg2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcpg2()
