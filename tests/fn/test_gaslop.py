"""gaslop is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gaslop import gaslop


def test_gaslop_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gaslop()
