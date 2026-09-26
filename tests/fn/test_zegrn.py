"""zegrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zegrn import gradient_spatial


def test_zegrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gradient_spatial(data=None)
