"""vtnet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtnet import vtnet


def test_vtnet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtnet()
