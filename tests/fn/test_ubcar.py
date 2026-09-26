"""ubcar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcar import ubcar


def test_ubcar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcar()
