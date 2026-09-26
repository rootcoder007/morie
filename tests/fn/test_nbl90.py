"""nbl90 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbl90 import nbl90


def test_nbl90_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbl90()
