"""nbl50 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbl50 import nbl50


def test_nbl50_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbl50()
