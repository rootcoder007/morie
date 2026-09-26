"""whtinv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.whtinv import walsh_hadamard_inverse


def test_whtinv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        walsh_hadamard_inverse(x=None)
