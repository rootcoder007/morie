"""vdedge is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdedge import vdedge


def test_vdedge_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdedge()
