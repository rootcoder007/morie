"""vgcub is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgcub import vario_cubic


def test_vgcub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_cubic(coords=None, values=None)
