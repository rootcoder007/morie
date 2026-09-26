"""matrwl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.matrwl import matrwl


def test_matrwl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        matrwl()
