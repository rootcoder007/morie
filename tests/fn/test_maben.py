"""maben is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maben import maben


def test_maben_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maben()
