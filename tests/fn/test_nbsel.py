"""nbsel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nbsel import nbsel


def test_nbsel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nbsel()
