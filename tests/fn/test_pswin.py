"""pswin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pswin import pswin


def test_pswin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pswin()
