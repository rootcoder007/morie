"""mgwrtvl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mgwrtvl import mgwrtvl


def test_mgwrtvl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mgwrtvl(coef=None, se=None)
