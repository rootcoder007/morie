"""vgmat is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgmat import vario_matern


def test_vgmat_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_matern(coords=None, values=None)
