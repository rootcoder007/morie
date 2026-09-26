"""rsreg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsreg import rsreg


def test_rsreg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsreg()
