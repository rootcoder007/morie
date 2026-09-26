"""vgns2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgns2 import vario_nested_fit


def test_vgns2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_nested_fit(coords=None, values=None)
