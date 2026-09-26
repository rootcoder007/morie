"""abfng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abfng import abfng


def test_abfng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abfng()
