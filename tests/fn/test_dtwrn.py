"""dtwrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtwrn import dtwrn


def test_dtwrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtwrn()
