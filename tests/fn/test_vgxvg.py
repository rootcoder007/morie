"""vgxvg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgxvg import cross_vario


def test_vgxvg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cross_vario(coords=None, values=None)
