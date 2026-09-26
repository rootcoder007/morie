"""dkstv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkstv import dkstv


def test_dkstv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkstv()
