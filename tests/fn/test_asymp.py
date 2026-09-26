"""asymp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.asymp import asymptotic_expansion


def test_asymp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        asymptotic_expansion(f=None, x_inf=None)
