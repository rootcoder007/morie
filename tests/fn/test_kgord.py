"""kgord is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgord import ordinary_kriging


def test_kgord_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ordinary_kriging(values=None, x=None)
