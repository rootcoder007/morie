"""eladv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.eladv import eladv


def test_eladv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        eladv()
