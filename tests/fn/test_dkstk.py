"""dkstk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkstk import dkstk


def test_dkstk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkstk()
