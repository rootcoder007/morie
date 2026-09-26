"""gethl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gethl import gethl


def test_gethl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gethl()
