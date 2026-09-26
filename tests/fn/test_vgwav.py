"""vgwav is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgwav import vario_wave


def test_vgwav_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_wave(coords=None, values=None)
