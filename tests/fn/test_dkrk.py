"""dkrk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkrk import dkrk


def test_dkrk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkrk()
