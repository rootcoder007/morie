"""secop is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secop import secop


def test_secop_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secop()
