"""mabuoy is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mabuoy import mabuoy


def test_mabuoy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mabuoy()
