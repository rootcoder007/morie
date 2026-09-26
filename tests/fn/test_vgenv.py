"""vgenv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgenv import vario_envelope


def test_vgenv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_envelope(coords=None, values=None)
