"""vgpen is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgpen import vario_pentaspherical


def test_vgpen_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_pentaspherical(coords=None, values=None)
