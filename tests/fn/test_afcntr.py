"""afcntr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afcntr import afcntr


def test_afcntr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afcntr()
