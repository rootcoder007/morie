"""spccir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spccir import spccir


def test_spccir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spccir()
