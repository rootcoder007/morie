"""ubsch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubsch import ubsch


def test_ubsch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubsch()
