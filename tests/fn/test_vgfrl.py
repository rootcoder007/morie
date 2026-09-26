"""vgfrl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgfrl import vario_fit_reml


def test_vgfrl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_fit_reml(coords=None, values=None)
