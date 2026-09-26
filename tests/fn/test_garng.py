"""garng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.garng import garng


def test_garng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        garng()
