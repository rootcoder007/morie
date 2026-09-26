"""geseg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.geseg import geseg


def test_geseg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        geseg()
