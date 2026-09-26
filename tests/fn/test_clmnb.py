"""clmnb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clmnb import clmnb


def test_clmnb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clmnb()
