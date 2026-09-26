"""laplI is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.laplI import inverse_laplace


def test_laplI_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        inverse_laplace(F=None, s=None, t=None)
