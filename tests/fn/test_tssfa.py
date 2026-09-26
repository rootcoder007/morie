"""tssfa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tssfa import tssfa


def test_tssfa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tssfa()
