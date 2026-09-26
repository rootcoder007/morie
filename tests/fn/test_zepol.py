"""zepol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zepol import pollution_surface


def test_zepol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pollution_surface(data=None)
