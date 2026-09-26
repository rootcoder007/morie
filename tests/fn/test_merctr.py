"""merctr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.merctr import merctr


def test_merctr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        merctr()
