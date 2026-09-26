"""kguni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kguni import universal_kriging


def test_kguni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        universal_kriging(values=None, x=None)
