"""ubwlk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubwlk import ubwlk


def test_ubwlk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubwlk()
