"""sesir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sesir import sesir


def test_sesir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sesir()
