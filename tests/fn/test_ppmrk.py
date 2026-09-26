"""ppmrk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmrk import ppmrk


def test_ppmrk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmrk()
