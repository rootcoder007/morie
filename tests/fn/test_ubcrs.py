"""ubcrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcrs import ubcrs


def test_ubcrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcrs()
