"""lmbp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lmbp import lmbp


def test_lmbp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lmbp(resid=None, X=None)
