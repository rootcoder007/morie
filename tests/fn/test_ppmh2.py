"""ppmh2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppmh2 import ppmh2


def test_ppmh2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppmh2()
