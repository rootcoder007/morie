"""vgaic is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgaic import vario_aic


def test_vgaic_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_aic(coords=None, values=None)
