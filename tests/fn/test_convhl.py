"""convhl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.convhl import convhl


def test_convhl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        convhl()
