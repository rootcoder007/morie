"""opgir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opgir import opgir


def test_opgir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opgir()
