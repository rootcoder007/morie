"""gcveg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcveg import gcveg


def test_gcveg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcveg()
