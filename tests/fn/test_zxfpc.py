"""zxfpc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxfpc import fpca_spatial


def test_zxfpc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fpca_spatial(data=None)
