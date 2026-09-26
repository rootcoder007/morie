"""kgind is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgind import indicator_kriging


def test_kgind_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        indicator_kriging(values=None, x=None)
