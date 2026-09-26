"""abhch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abhch import abhch


def test_abhch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abhch()
