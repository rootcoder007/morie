"""vgpow is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgpow import vario_power


def test_vgpow_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_power(coords=None, values=None)
