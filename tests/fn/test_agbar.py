"""agbar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agbar import agbar


def test_agbar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agbar()
