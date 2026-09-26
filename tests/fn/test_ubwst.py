"""ubwst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubwst import ubwst


def test_ubwst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubwst()
