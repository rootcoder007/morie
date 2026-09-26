"""sfpve is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sfpve import sfpve


def test_sfpve_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sfpve(y=None, evecs=None)
