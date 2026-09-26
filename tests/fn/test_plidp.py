"""plidp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plidp import plidp


def test_plidp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plidp()
