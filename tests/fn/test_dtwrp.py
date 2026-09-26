"""dtwrp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtwrp import dtwrp


def test_dtwrp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtwrp()
