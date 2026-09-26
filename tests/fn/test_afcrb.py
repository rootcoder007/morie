"""afcrb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afcrb import afcrb


def test_afcrb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afcrb()
