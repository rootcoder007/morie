"""vgros is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgros import vario_rose


def test_vgros_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_rose(coords=None, values=None)
