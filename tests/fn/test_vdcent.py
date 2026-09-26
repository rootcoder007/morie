"""vdcent is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdcent import vdcent


def test_vdcent_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdcent()
