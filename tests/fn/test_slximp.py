"""slximp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slximp import slximp


def test_slximp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slximp(coef=None, theta=None)
