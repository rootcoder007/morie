"""ppmov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmov import ppmov


def test_ppmov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmov()
