"""zxnbt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxnbt import network_between


def test_zxnbt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        network_between(data=None)
