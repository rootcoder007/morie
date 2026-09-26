"""vgcir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgcir import vario_circular


def test_vgcir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_circular(coords=None, values=None)
