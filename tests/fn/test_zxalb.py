"""zxalb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxalb import albers_proj


def test_zxalb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        albers_proj(data=None)
