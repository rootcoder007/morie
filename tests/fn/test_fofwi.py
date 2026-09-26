"""fofwi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fofwi import fofwi


def test_fofwi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fofwi()
