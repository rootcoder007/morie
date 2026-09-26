"""abvir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abvir import abvir


def test_abvir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abvir()
