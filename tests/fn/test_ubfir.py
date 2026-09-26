"""ubfir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubfir import ubfir


def test_ubfir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubfir()
