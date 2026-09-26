"""csdkp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csdkp import csdkp


def test_csdkp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csdkp()
