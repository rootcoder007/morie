"""nrstk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nrstk import nrstk


def test_nrstk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nrstk()
