"""hscir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hscir import hscir


def test_hscir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hscir()
