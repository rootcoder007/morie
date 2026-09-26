"""lmhe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lmhe import lmhe


def test_lmhe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lmhe(resid=None, X=None)
