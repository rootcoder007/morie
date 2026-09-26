"""lmkoenk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lmkoenk import lmkoenk


def test_lmkoenk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lmkoenk(resid=None, X=None)
