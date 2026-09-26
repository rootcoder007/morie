"""zsmci is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsmci import mc_spatial_int


def test_zsmci_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mc_spatial_int(data=None)
