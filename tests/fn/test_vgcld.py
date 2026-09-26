"""vgcld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgcld import vario_cloud


def test_vgcld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_cloud(coords=None, values=None)
