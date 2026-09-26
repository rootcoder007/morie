"""dkgnk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkgnk import dkgnk


def test_dkgnk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkgnk()
