"""ppmhr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmhr import ppmhr


def test_ppmhr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmhr()
