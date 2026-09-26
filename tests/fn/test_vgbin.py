"""vgbin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgbin import vario_binned


def test_vgbin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_binned(coords=None, values=None)
