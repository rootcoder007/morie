"""kgfct is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgfct import factorial_kriging


def test_kgfct_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        factorial_kriging(values=None, x=None)
