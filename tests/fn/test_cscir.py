"""cscir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cscir import cscir


def test_cscir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cscir()
