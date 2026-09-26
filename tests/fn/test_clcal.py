"""clcal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clcal import clcal


def test_clcal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clcal()
