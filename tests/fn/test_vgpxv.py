"""vgpxv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgpxv import pseudo_cross_vario


def test_vgpxv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pseudo_cross_vario(coords=None, values=None)
