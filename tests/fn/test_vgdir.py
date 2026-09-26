"""vgdir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgdir import directional_vario


def test_vgdir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        directional_vario(coords=None, values=None)
