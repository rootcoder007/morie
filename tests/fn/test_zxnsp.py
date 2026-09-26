"""zxnsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxnsp import network_shortest


def test_zxnsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        network_shortest(data=None)
