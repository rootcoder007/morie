"""vgsph is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vgsph import vario_spherical


def test_vgsph_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vario_spherical(coords=None, values=None)
