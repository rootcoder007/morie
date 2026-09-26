"""sfsel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfsel import sfsel


def test_sfsel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfsel(y=None, X=None, W=None)
