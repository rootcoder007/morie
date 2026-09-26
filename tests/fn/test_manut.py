"""manut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.manut import manut


def test_manut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        manut()
