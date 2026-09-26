"""dtcir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcir import dtcir


def test_dtcir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcir()
